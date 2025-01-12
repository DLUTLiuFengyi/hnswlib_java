package com.github.jelmerk.knn.hnsw;


import com.github.jelmerk.knn.*;
import com.github.jelmerk.knn.util.*;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.api.map.primitive.MutableObjectIntMap;
import org.eclipse.collections.api.map.primitive.MutableObjectLongMap;
import org.eclipse.collections.api.tuple.primitive.ObjectIntPair;
import org.eclipse.collections.api.tuple.primitive.ObjectLongPair;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectLongHashMap;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.*;

/**
 * Implementation of {@link Index} that implements the hnsw algorithm.
 *
 * @param <TId>       Type of the external identifier of an item
 * @param <TVector>   Type of the vector to perform distance calculation on
 * @param <TItem>     Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 * @see <a href="https://arxiv.org/abs/1603.09320">
 * Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a>
 */
public class HnswIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        implements Index<TId, TVector, TItem, TDistance> {

    private static final byte VERSION_1 = 0x01;
    private static final byte VERSION_2 = 0x02;

    private static final long serialVersionUID = 1L;

    private static final int NO_NODE_ID = -1;

    private DistanceFunction<TVector, TDistance> distanceFunction;
    private Comparator<TDistance> distanceComparator;
    private MaxValueComparator<TDistance> maxValueDistanceComparator;

    private boolean immutable;
    private int dimensions;
    private int maxItemCount;
    private int m;
    private int maxM;
    private int maxM0;
    private double levelLambda;
    private int ef;
    private int efConstruction;
    private boolean removeEnabled;

    private int nodeCount;

    private volatile Node<TItem> entryPoint;

    private AtomicReferenceArray<Node<TItem>> nodes;
    private MutableObjectIntMap<TId> lookup;
    private MutableObjectLongMap<TId> deletedItemVersions;
    private Map<TId, Object> locks;

    private ObjectSerializer<TId> itemIdSerializer;
    private ObjectSerializer<TItem> itemSerializer;

    private ReentrantLock globalLock;

    private GenericObjectPool<ArrayBitSet> visitedBitSetPool;

    private ArrayBitSet excludedCandidates;

    private ExactView exactView;

    private HnswIndex(RefinedBuilder<TId, TVector, TItem, TDistance> builder) {

        this.immutable = builder.immutable;
        this.dimensions = builder.dimensions;
        this.maxItemCount = builder.maxItemCount;
        this.distanceFunction = builder.distanceFunction;
        this.distanceComparator = builder.distanceComparator;
        this.maxValueDistanceComparator = new MaxValueComparator<>(this.distanceComparator);

        this.m = builder.m;
        this.maxM = builder.m;
        this.maxM0 = builder.m * 2;
        this.levelLambda = 1 / Math.log(this.m);
        this.efConstruction = Math.max(builder.efConstruction, m);
        this.ef = builder.ef;
        this.removeEnabled = builder.removeEnabled;

        this.nodes = new AtomicReferenceArray<>(this.maxItemCount);

        this.lookup = new ObjectIntHashMap<>();
        this.deletedItemVersions = new ObjectLongHashMap<>();
        this.locks = new HashMap<>();

        this.itemIdSerializer = builder.itemIdSerializer;
        this.itemSerializer = builder.itemSerializer;

        this.globalLock = new ReentrantLock();

        this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount),
                Runtime.getRuntime().availableProcessors());

        this.excludedCandidates = new ArrayBitSet(this.maxItemCount);

        this.exactView = new ExactView();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        globalLock.lock();
        try {
            return lookup.size();
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Optional<TItem> get(TId id) {
        globalLock.lock();
        try {
            int nodeId = lookup.getIfAbsent(id, NO_NODE_ID);

            if (nodeId == NO_NODE_ID) {
                return Optional.empty();
            } else {
                return Optional.of(nodes.get(nodeId).item);
            }
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<TItem> items() {
        globalLock.lock();
        try {
            List<TItem> results = new ArrayList<>(size());

            Iterator<TItem> iter = new ItemIterator();

            while(iter.hasNext()) {
                results.add(iter.next());
            }

            return results;
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean remove(TId id, long version) {

        if (!removeEnabled) {
            return false;
        }

        globalLock.lock();

        try {
            int internalNodeId = lookup.getIfAbsent(id, NO_NODE_ID);

            if (internalNodeId == NO_NODE_ID) {
                return false;
            }

            Node<TItem> node = nodes.get(internalNodeId);

            if (version < node.item.version()) {
                return false;
            }

            node.deleted = true;

            lookup.remove(id);

            deletedItemVersions.put(id, version);

            return true;
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * 插入新节点，用于HNSW构建阶段
     * {@inheritDoc}
     * @param item the item to add to the index
     * @return
     */
    @Override
    public boolean add(TItem item) {
        if (immutable) {
            throw new UnsupportedOperationException("Index is immutable");
        }
        if (item.dimensions() != dimensions) {
            throw new IllegalArgumentException("Item does not have dimensionality of : " + dimensions);
        }
        // 获取该节点被分配到randomLevel层数
        int randomLevel = assignLevel(item.id(), this.levelLambda);
        // 每层都维护一份邻居列表，randomLevel,...,1,0
        IntArrayList[] connections = new IntArrayList[randomLevel + 1];
        // 对randomLevel,...,1,0，初始化该节点在每层的邻居节点集合
        for (int level = 0; level <= randomLevel; level++) {
            int levelM = randomLevel == 0 ? maxM0 : maxM; // 最底层的maxM值与其他层不同
            connections[level] = new IntArrayList(levelM);
        }
        // 整个插入过程，上锁
        globalLock.lock();
        try {
            // 如果插入节点的id在图中不存在，返回-1
            int existingNodeId = lookup.getIfAbsent(item.id(), NO_NODE_ID);

            if (existingNodeId != NO_NODE_ID) { // 如果插入节点在图中已存在
                if (!removeEnabled) {
                    return false;
                }
                // 根据id获取对应的真实节点
                Node<TItem> node = nodes.get(existingNodeId);
                if (item.version() < node.item.version()) {
                    return false;
                }
                // 节点对应的向量数据相同，无需执行插入操作
                if (Objects.deepEquals(node.item.vector(), item.vector())) {
                    node.item = item;
                    return true;
                } else { // 否则，先删除旧版本
                    remove(item.id(), item.version());
                }
            } else if (item.version() < deletedItemVersions.getIfAbsent(item.id(), -1)) {
                return false;
            }
            // 超过HNSW的最大节点数限制
            if (nodeCount >= this.maxItemCount) {
                throw new SizeLimitExceededException("The number of elements exceeds the specified limit.");
            }

            int newNodeId = nodeCount++;
            synchronized (excludedCandidates) {
                excludedCandidates.add(newNodeId);
            }
            // 创建新节点对象
            Node<TItem> newNode = new Node<>(newNodeId, connections, item, false);
            nodes.set(newNodeId, newNode);
            lookup.put(item.id(), newNodeId);
            deletedItemVersions.remove(item.id());

            Object lock = locks.computeIfAbsent(item.id(), k -> new Object());
            // HNSW的整体入口节点
            Node<TItem> entryPointCopy = entryPoint;

            try {
                synchronized (lock) {
                    synchronized (newNode) {
                        //
                        if (entryPoint != null && randomLevel <= entryPoint.maxLevel()) {
                            globalLock.unlock();
                        }

                        Node<TItem> currObj = entryPointCopy;
                        // 入口节点不为空
                        if (currObj != null) {
                            // 新节点被分配的层数小于入口节点的层数，才能执行插入
                            if (newNode.maxLevel() < entryPointCopy.maxLevel()) {
                                // 新节点item与当前节点cur的距离
                                TDistance curDist = distanceFunction.distance(item.vector(), currObj.item.vector());
                                // 遍历从顶层L到新节点被分配的最高层的上一层l+1，不断更新入口节点
                                for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > newNode.maxLevel(); activeLevel--) {
                                    boolean changed = true;
                                    while (changed) {
                                        changed = false;
                                        synchronized (currObj) {
                                            // 获取当前节点（入口点）cur的邻居集合
                                            MutableIntList candidateConnections = currObj.connections[activeLevel];
                                            // 遍历cur的邻居
                                            for (int i = 0; i < candidateConnections.size(); i++) {
                                                int candidateId = candidateConnections.get(i);
                                                Node<TItem> candidateNode = nodes.get(candidateId);
                                                // 计算cur的邻居与新节点的距离
                                                TDistance candidateDistance = distanceFunction.distance(
                                                        item.vector(),
                                                        candidateNode.item.vector()
                                                );
                                                // 如果cur节点的邻居与新节点的距离 小于 cur节点与新节点之间的距离
                                                if (lt(candidateDistance, curDist)) {
                                                    // 将cur节点换成该邻居
                                                    curDist = candidateDistance;
                                                    currObj = candidateNode;
                                                    changed = true; // 标志置为true，表示while继续循环，不断逼近与q更近的节点
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // 此时，cur节点是第l+1层中距离新节点最近的点，作为第l层的入口节点
                            // 对 l,...,1,0层，遍历每一层，开始执行插入，每一层都需插入该新节点
                            for (int level = Math.min(randomLevel, entryPointCopy.maxLevel()); level >= 0; level--) {
                                // 在当前层，查离新节点在这一层的最近的efConstruction个点（SEARCH-LAYER算法），赋值给W
                                PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates =
                                        searchBaseLayer(currObj, item.vector(), efConstruction, level);
                                // 将初始的入口点也加入结果集W（这步没明白）
                                if (entryPointCopy.deleted) {
                                    TDistance distance = distanceFunction.distance(item.vector(), entryPointCopy.item.vector());
                                    topCandidates.add(new NodeIdAndDistance<>(entryPointCopy.id, distance, maxValueDistanceComparator));
                                    if (topCandidates.size() > efConstruction) { // 如果加了入口点后W长度大于阈值，则弹出堆顶
                                        topCandidates.poll();
                                    }
                                }
                                // 执行插入新节点的相关操作：（1）给相关节点建立连接（2）收缩过多连接
                                mutuallyConnectNewElement(newNode, topCandidates, level);
                            }
                        }
                        // zoom out to the highest level
                        // 整个hnsw图的entryPoint定为第一个插入图的节点
                        // 如果后面有比它的maxLevel更高的新节点，则将entryPoint更新为那个新节点
                        if (entryPoint == null || newNode.maxLevel() > entryPointCopy.maxLevel()) {
                            // this is thread safe because we get the global lock when we add a level
                            this.entryPoint = newNode;
                        }
                        return true;
                    }
                }
            } finally {
                synchronized (excludedCandidates) {
                    excludedCandidates.remove(newNodeId);
                }
            }
        } finally {
            if (globalLock.isHeldByCurrentThread()) {
                globalLock.unlock();
            }
        }
    }

    /**
     * 执行插入新节点的相关操作，包括给相关点建立新连接、收缩过多连接
     *
     * @param newNode 要插入的新节点
     * @param topCandidates 该节点的topK集合，是最大堆
     * @param level 当前层数
     */
    private void mutuallyConnectNewElement(Node<TItem> newNode,
                                           PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates,
                                           int level) {
        // 最底层的maxM与其他层不一样
        int bestN = level == 0 ? this.maxM0 : this.maxM;
        int newNodeId = newNode.id;
        TVector newItemVector = newNode.item.vector();
        MutableIntList newItemConnections = newNode.connections[level]; // 插入节点与其邻居的连接的集合
        // 在当前层，基于给定的topK集合W，查找离插入节点最近的m个点（W的长度efConstruction大于m）（SELECT-NEIGHBORS算法）
        // 启发式算法，扩大遍历范围，访问topK集合里的点的邻居节点，获取到新的topK集合
        getNeighborsByHeuristic2(topCandidates, m);
        // 遍历新的topK集合，即插入点q的邻居
        while (!topCandidates.isEmpty()) {
            int selectedNeighbourId = topCandidates.poll().nodeId; // 堆顶是邻居集合里距离插入点q最远的节点e

            synchronized (excludedCandidates) {
                if (excludedCandidates.contains(selectedNeighbourId)) {
                    continue;
                }
            }
            // 给插入点q和其邻居节点e建立连接
            newItemConnections.add(selectedNeighbourId);
            // 获取邻居的真实节点对象
            Node<TItem> neighbourNode = nodes.get(selectedNeighbourId);
            // 检查该邻居节点e的连接数是否超过Mmax，如果超过，则需进行收缩
            synchronized (neighbourNode) {
                TVector neighbourVector = neighbourNode.item.vector();
                // 获取节点e与其邻居的连接集合
                MutableIntList neighbourConnectionsAtLevel = neighbourNode.connections[level];
                // 如果节点e的连接数尚未超过预设值bestN（即论文里的Mmax），表示可以安全地建立其与插入点q的连接
                if (neighbourConnectionsAtLevel.size() < bestN) {
                    neighbourConnectionsAtLevel.add(newNodeId);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    // 否则，将最远的节点剔除出节点e的连接集合
                    // 先定义插入节点q与节点e的距离
                    TDistance dMax = distanceFunction.distance(
                            newItemVector,
                            neighbourNode.item.vector()
                    );
                    Comparator<NodeIdAndDistance<TDistance>> comparator = Comparator
                            .<NodeIdAndDistance<TDistance>>naturalOrder().reversed();
                    // 建立候选集合，是最大堆，堆顶是该堆里距离插入节点q最远的节点
                    PriorityQueue<NodeIdAndDistance<TDistance>> candidates = new PriorityQueue<>(comparator);
                    candidates.add(new NodeIdAndDistance<>(newNodeId, dMax, maxValueDistanceComparator));
                    // 对节点e的每一个邻居，计算它们与节点q的距离，添加到最大堆中
                    neighbourConnectionsAtLevel.forEach(id -> {
                        TDistance dist = distanceFunction.distance(
                                neighbourVector,
                                nodes.get(id).item.vector()
                        );
                        candidates.add(new NodeIdAndDistance<>(id, dist, maxValueDistanceComparator));
                    });
                    // 启发式算法，访问该最大堆里每个点的邻居节点，获取到新的最大堆，这一步目的是扩大遍历范围，尽可能多地考虑图中的其它节点
                    // 将新的最大堆的长度限制为bestN
                    getNeighborsByHeuristic2(candidates, bestN);
                    neighbourConnectionsAtLevel.clear();
                    // 重新设置节点e的对外连接集合
                    while (!candidates.isEmpty()) {
                        neighbourConnectionsAtLevel.add(candidates.poll().nodeId);
                    }
                }
            }
        }
    }

    /**
     * 启发式方法，基于集合C查找距离给定节点q最近的m个邻居
     * 该方法是影响性能和准确率的关键之一
     *
     * @param topCandidates 候选集合C，是最大堆，堆顶是C中距离节点q最远的节点
     * @param m 要返回的结果集的长度
     */
    private void getNeighborsByHeuristic2(PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates, int m) {
        if (topCandidates.size() < m) {
            return;
        }
        // 工作队列W，是最小堆
        PriorityQueue<NodeIdAndDistance<TDistance>> queueClosest = new PriorityQueue<>();
        List<NodeIdAndDistance<TDistance>> returnList = new ArrayList<>(); // 结果集，即论文里的R
        // 将C中所有节点放进工作队列W
        while(!topCandidates.isEmpty()) {
            queueClosest.add(topCandidates.poll());
        }
        // 遍历W
        while(!queueClosest.isEmpty()) {
            if (returnList.size() >= m) { // 结果已凑齐，结束循环
                break;
            }
            // 弹出最小堆的堆顶，目前距离q最近的节点
            NodeIdAndDistance<TDistance> currentPair = queueClosest.poll();
            TDistance distToQuery = currentPair.distance;
            // good=true表示当前遍历到的节点cur可以安全添加进最后结果集中
            boolean good = true;
            for (NodeIdAndDistance<TDistance> secondPair : returnList) {
                // 遍历此时的结果集，计算工作队列W弹出的堆顶cur节点和结果集内每一个节点之间的距离
                TDistance curdist = distanceFunction.distance(
                        nodes.get(secondPair.nodeId).item.vector(),
                        nodes.get(currentPair.nodeId).item.vector()
                );
                // 如果 cur节点和结果集遍历到的second节点之间的距离 ＜ cur节点和插入点q之间的距离
                // 说明cur节点可能不是离插入点q最近的节点，将good设为false
                if (lt(curdist, distToQuery)) {
                    good = false;
                    break;
                }
            }
            // for循环结束后如果good=false，则判断cur节点不是离插入点q最近的节点，有更好的选择
            if (good) { // 如果good=true，则判断为可将cur节点添加进结果集
                returnList.add(currentPair);
            }
        }
        // 将新的结果集赋值给插入点q的候选集合C
        topCandidates.addAll(returnList);
    }

    /**
     * 基于HNSW索引的TopK查询
     * * {@inheritDoc}
     * @param destination 查询向量 q
     * @param k number of items to return
     * @return q的topK查询结果
     */
    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector destination, int k) {
        if (entryPoint == null) {
            return Collections.emptyList();
        }
        // entryPoint在hnsw图中已提前设定好
        Node<TItem> entryPointCopy = entryPoint;
        // 当前遍历节点
        Node<TItem> currObj = entryPointCopy;
        // 查询变量与入口节点的距离
        TDistance curDist = distanceFunction.distance(destination, currObj.item.vector());
        // 从顶层往下，到倒数第2层，每层找当前层q最近邻的ef=1个点，赋值到集合W
        for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > 0; activeLevel--) {
            boolean changed = true;
            while (changed) {
                changed = false;
                synchronized (currObj) {
                    // 获取当前节点在该层的邻居节点
                    MutableIntList candidateConnections = currObj.connections[activeLevel];
                    // 遍历邻居节点
                    for (int i = 0; i < candidateConnections.size(); i++) {
                        // 获取该邻居节点的id
                        int candidateId = candidateConnections.get(i);
                        // nodes存储的是所有节点对象，通过id获取当前遍历到的邻居节点，计算查询节点与邻居节点的向量之间的距离
                        TDistance candidateDistance = distanceFunction.distance(
                                destination,
                                nodes.get(candidateId).item.vector()
                        );
                        // 如果查询q与cur的邻居节点的距离 小于 查询q与当前节点cur之间的距离
                        if (lt(candidateDistance, curDist)) {
                            // 当前节点cur更新为该邻居节点
                            curDist = candidateDistance;
                            currObj = nodes.get(candidateId);
                            // 改变标志置为true，继续进行循环
                            changed = true;
                        }
                        // 继续判断原当前节点的邻居们
                        // 得先走完这个for循环，将原当前节点的所有邻居都判断完，才会跳到原当前节点的所有邻居中距离q最近的节点
                        // 再进行下一次邻居节点的for循环判断
                    }
                }
               // 如果改变标志为false，说明当前节点cur已经是这一层所有节点中离q最近的节点，循环中止
            }
        }
        // 上一个while循环结束，currObj已经是倒数第二层中距离查询点q最近的节点，成为最底层的入口节点
        // 在最底层进行knn查询，k取Max(ef,k)
        // 返回一个优先队列，是最大堆，堆顶为候选结果集中距离q最远的节点
        PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates = searchBaseLayer(
                currObj, destination, Math.max(ef, k), 0);
        // 取topK
        while (topCandidates.size() > k) {
            topCandidates.poll();
        }
        // 结果封装
        List<SearchResult<TItem, TDistance>> results = new ArrayList<>(topCandidates.size());
        while (!topCandidates.isEmpty()) {
            NodeIdAndDistance<TDistance> pair = topCandidates.poll();
            results.add(0, new SearchResult<>(nodes.get(pair.nodeId).item, pair.distance, maxValueDistanceComparator));
        }
        return results;
    }

    /**
     * Changes the maximum capacity of the index.
     * @param newSize new size of the index
     */
    public void resize(int newSize) {
        globalLock.lock();
        try {
            this.maxItemCount = newSize;

            this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount),
                    Runtime.getRuntime().availableProcessors());

            AtomicReferenceArray<Node<TItem>> newNodes = new AtomicReferenceArray<>(newSize);
            for(int i = 0; i < this.nodes.length(); i++) {
                newNodes.set(i, this.nodes.get(i));
            }
            this.nodes = newNodes;

            this.excludedCandidates = new ArrayBitSet(this.excludedCandidates, newSize);
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * 在当前层进行TopK查询
     *
     * @param entryPointNode 入口节点ep
     * @param destination 查询向量q
     * @param k
     * @param layer 第几层
     * @return 一个最大堆，堆顶为候选结果集中距离查询向量q最远的节点
     */
    private PriorityQueue<NodeIdAndDistance<TDistance>> searchBaseLayer(
            Node<TItem> entryPointNode, TVector destination, int k, int layer) {
        // visited集合
        ArrayBitSet visitedBitSet = visitedBitSetPool.borrowObject();
        try {
            // q的最近邻集合，即论文里的W，是最大堆，堆顶是集合里距离查询点q距离最大的节点
            PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates =
                    new PriorityQueue<>(Comparator.<NodeIdAndDistance<TDistance>>naturalOrder().reversed());
            // 候选节点集合，即论文里的C，是最小堆，堆顶是集合里距离查询点q距离最小的节点
            PriorityQueue<NodeIdAndDistance<TDistance>> candidateSet = new PriorityQueue<>();
            TDistance lowerBound;
            if (!entryPointNode.deleted) { // 结果集中是否考虑入口点
                TDistance distance = distanceFunction.distance(destination, entryPointNode.item.vector());
                NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, distance, maxValueDistanceComparator);
                // W ⬅ ep
                topCandidates.add(pair);
                lowerBound = distance; // 距离边界为入口点与查询点q的距离
                // C ⬅ ep
                candidateSet.add(pair);
            } else {
                lowerBound = MaxValueComparator.maxValue();
                NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, lowerBound, maxValueDistanceComparator);
                candidateSet.add(pair);
            }
            // v ⬅ ep
            visitedBitSet.add(entryPointNode.id);
            // 当C不为空
            while (!candidateSet.isEmpty()) {
                // 从C中取出距离查询点q最近的元素cur
                NodeIdAndDistance<TDistance> currentPair = candidateSet.poll();
                // lowerBound是结果集合W中距离q最远的元素f与q的距离
                if (gt(currentPair.distance, lowerBound)) { // 如果(cur,q)的距离大于(f,q)的距离，循环中止，表示已找到所有topK
                    break;
                }
                // C存储的是<id,节点与q的距离>，这里根据id，从HNSW图中获取到cur的真正节点对象
                Node<TItem> node = nodes.get(currentPair.nodeId);
                // 下面要访问HNSW图中的真实节点，因此上锁
                synchronized (node) {
                    // 获取cur节点在该层的邻居
                    MutableIntList candidates = node.connections[layer];
                    // for循环遍历cur的邻居
                    for (int i = 0; i < candidates.size(); i++) {
                        int candidateId = candidates.get(i);
                        if (!visitedBitSet.contains(candidateId)) { // 该邻居节点没被评估过，进行评估
                            // 将当前邻居节点e添加进visited
                            visitedBitSet.add(candidateId);
                            // 根据id获取邻居节点的真正对象
                            Node<TItem> candidateNode = nodes.get(candidateId);
                            // 计算其与q的距离
                            TDistance candidateDistance = distanceFunction.distance(destination,
                                    candidateNode.item.vector());
                            // lowerBound是W中距离q最远的节点与q之间的距离
                            // 如果 W中元素个数＜k 或者 lowerBound＞(e,q)的距离
                            if (topCandidates.size() < k || gt(lowerBound, candidateDistance)) {
                                // 将该邻居节点e添加进候选集C中
                                NodeIdAndDistance<TDistance> candidatePair =
                                        new NodeIdAndDistance<>(candidateId, candidateDistance, maxValueDistanceComparator);
                                candidateSet.add(candidatePair);
                                if (!candidateNode.deleted) {
                                    // 将该邻居节点e添加进结果集W中
                                    topCandidates.add(candidatePair);
                                }
                                // 控制W个数，将堆顶弹出
                                if (topCandidates.size() > k) {
                                    topCandidates.poll();
                                }
                                // 更新lowerBound为新的W中距离q最远的节点与q的距离
                                if (!topCandidates.isEmpty()) {
                                    lowerBound = topCandidates.peek().distance;
                                }
                            }
                        }
                    }

                }
            }
            return topCandidates;
        } finally {
            visitedBitSet.clear();
            visitedBitSetPool.returnObject(visitedBitSet);
        }
    }

    /**
     * Creates a read only view on top of this index that uses pairwise comparision when doing distance search. And as
     * such can be used as a baseline for assessing the precision of the index.
     * Searches will be really slow but give the correct result every time.
     *
     * @return read only view on top of this index that uses pairwise comparision when doing distance search
     */
    public Index<TId, TVector, TItem, TDistance> asExactIndex() {
        return exactView;
    }

    /**
     * Returns the dimensionality of the items stored in this index.
     *
     * @return the dimensionality of the items stored in this index
     */
    public int getDimensions() {
        return dimensions;
    }

    /**
     * Returns the number of bi-directional links created for every new element during construction.
     *
     * @return the number of bi-directional links created for every new element during construction
     */
    public int getM() {
        return m;
    }

    /**
     * The size of the dynamic list for the nearest neighbors (used during the search)
     *
     * @return The size of the dynamic list for the nearest neighbors
     */
    public int getEf() {
        return ef;
    }

    /**
     * Set the size of the dynamic list for the nearest neighbors (used during the search)
     *
     * @param ef The size of the dynamic list for the nearest neighbors
     */
    public void setEf(int ef) {
        this.ef = ef;
    }

    /**
     * Returns the parameter has the same meaning as ef, but controls the index time / index precision.
     *
     * @return the parameter has the same meaning as ef, but controls the index time / index precision
     */
    public int getEfConstruction() {
        return efConstruction;
    }

    /**
     * Returns the distance function.
     *
     * @return the distance function
     */
    public DistanceFunction<TVector, TDistance> getDistanceFunction() {
        return distanceFunction;
    }


    /**
     * Returns the comparator used to compare distances.
     *
     * @return the comparator used to compare distance
     */
    public Comparator<TDistance> getDistanceComparator() {
        return distanceComparator;
    }

    /**
     * Returns if removes are enabled.
     *
     * @return true if removes are enabled for this index.
     */
    public boolean isRemoveEnabled() {
        return removeEnabled;
    }

    /**
     * Returns the maximum number of items the index can hold.
     *
     * @return the maximum number of items the index can hold
     */
    public int getMaxItemCount() {
        return maxItemCount;
    }

    /**
     * Returns the serializer used to serialize item id's when saving the index.
     *
     * @return the serializer used to serialize item id's when saving the index
     */
    public ObjectSerializer<TId> getItemIdSerializer() {
        return itemIdSerializer;
    }

    /**
     * Returns the serializer used to serialize items when saving the index.
     *
     * @return the serializer used to serialize items when saving the index
     */
    public ObjectSerializer<TItem> getItemSerializer() {
        return itemSerializer;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void save(OutputStream out) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.writeByte(VERSION_2);
        oos.writeInt(dimensions);
        oos.writeObject(distanceFunction);
        oos.writeObject(distanceComparator);
        oos.writeObject(itemIdSerializer);
        oos.writeObject(itemSerializer);
        oos.writeInt(maxItemCount);
        oos.writeInt(m);
        oos.writeInt(maxM);
        oos.writeInt(maxM0);
        oos.writeDouble(levelLambda);
        oos.writeInt(ef);
        oos.writeInt(efConstruction);
        oos.writeBoolean(removeEnabled);
        oos.writeInt(nodeCount);
        writeMutableObjectIntMap(oos, lookup);
        writeMutableObjectLongMap(oos, deletedItemVersions);
        writeNodesArray(oos, nodes);
        oos.writeInt(entryPoint == null ? -1 : entryPoint.id);
        oos.writeBoolean(immutable);
    }

    @SuppressWarnings("unchecked")
    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        @SuppressWarnings("unused") byte version = ois.readByte(); // for coping with future incompatible serialization
        this.dimensions = ois.readInt();
        this.distanceFunction = (DistanceFunction<TVector, TDistance>) ois.readObject();
        this.distanceComparator = (Comparator<TDistance>) ois.readObject();
        this.maxValueDistanceComparator = new MaxValueComparator<>(distanceComparator);
        this.itemIdSerializer = (ObjectSerializer<TId>) ois.readObject();
        this.itemSerializer = (ObjectSerializer<TItem>) ois.readObject();

        this.maxItemCount = ois.readInt();
        this.m = ois.readInt();
        this.maxM = ois.readInt();
        this.maxM0 = ois.readInt();
        this.levelLambda = ois.readDouble();
        this.ef = ois.readInt();
        this.efConstruction = ois.readInt();
        this.removeEnabled = ois.readBoolean();
        this.nodeCount = ois.readInt();
        this.lookup = readMutableObjectIntMap(ois, itemIdSerializer);
        this.deletedItemVersions = readMutableObjectLongMap(ois, itemIdSerializer);
        this.nodes = readNodesArray(ois, itemSerializer, maxM0, maxM);

        int entrypointNodeId = ois.readInt();

        this.immutable = version != VERSION_1 && ois.readBoolean();
        this.entryPoint = entrypointNodeId == -1 ? null : nodes.get(entrypointNodeId);

        this.globalLock = new ReentrantLock();
        this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount),
                Runtime.getRuntime().availableProcessors());
        this.excludedCandidates = new ArrayBitSet(this.maxItemCount);
        this.locks = new HashMap<>();
        this.exactView = new ExactView();
    }

    private void writeMutableObjectIntMap(ObjectOutputStream oos, MutableObjectIntMap<TId> map) throws IOException {
        oos.writeInt(map.size());

        for (ObjectIntPair<TId> pair : map.keyValuesView()) {
            itemIdSerializer.write(pair.getOne(), oos);
            oos.writeInt(pair.getTwo());
        }
    }

    private void writeMutableObjectLongMap(ObjectOutputStream oos, MutableObjectLongMap<TId> map) throws IOException {
        oos.writeInt(map.size());

        for (ObjectLongPair<TId> pair : map.keyValuesView()) {
            itemIdSerializer.write(pair.getOne(), oos);
            oos.writeLong(pair.getTwo());
        }
    }

    private void writeNodesArray(ObjectOutputStream oos, AtomicReferenceArray<Node<TItem>> nodes) throws IOException {
        oos.writeInt(nodes.length());
        for (int i = 0; i < nodes.length(); i++) {
            writeNode(oos, nodes.get(i));
        }
    }

    private void writeNode(ObjectOutputStream oos, Node<TItem> node) throws IOException {
        if (node == null) {
            oos.writeInt(-1);
        } else {
            oos.writeInt(node.id);
            oos.writeInt(node.connections.length);

            for (MutableIntList connections : node.connections) {
                oos.writeInt(connections.size());
                for (int j = 0; j < connections.size(); j++) {
                    oos.writeInt(connections.get(j));
                }
            }
            itemSerializer.write(node.item, oos);
            oos.writeBoolean(node.deleted);
        }
    }

    /**
     * Restores a {@link HnswIndex} from a File.
     *
     * @param file        File to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(File file)
            throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link HnswIndex} from a File.
     *
     * @param file        File to restore the index from
     * @param classLoader the classloader to use
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(File file, ClassLoader classLoader)
            throws IOException {
        return load(new FileInputStream(file), classLoader);
    }

    /**
     * Restores a {@link HnswIndex} from a Path.
     *
     * @param path        Path to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(Path path)
            throws IOException {
        return load(Files.newInputStream(path));
    }

    /**
     * Restores a {@link HnswIndex} from a Path.
     *
     * @param path        Path to restore the index from
     * @param classLoader the classloader to use
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     *
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(Path path, ClassLoader classLoader)
            throws IOException {
        return load(Files.newInputStream(path), classLoader);
    }

    /**
     * Restores a {@link HnswIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ...).
     * @return The restored index
     * @throws IOException              in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream)
            throws IOException {
        return load(inputStream, Thread.currentThread().getContextClassLoader());
    }

    /**
     * Restores a {@link HnswIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     * @param classLoader the classloader to use
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ...).
     * @return The restored index
     * @throws IOException              in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream, ClassLoader classLoader)
            throws IOException {

        try(ObjectInputStream ois = new ClassLoaderObjectInputStream(classLoader, inputStream)) {
            return (HnswIndex<TId, TVector, TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }

    private static IntArrayList readIntArrayList(ObjectInputStream ois, int initialSize) throws IOException {
        int size = ois.readInt();

        IntArrayList list = new IntArrayList(initialSize);

        for (int j = 0; j < size; j++) {
            list.add(ois.readInt());
        }

        return list;
    }

    private static <TItem> Node<TItem> readNode(ObjectInputStream ois,
                                                ObjectSerializer<TItem> itemSerializer,
                                                int maxM0,
                                                int maxM) throws IOException, ClassNotFoundException {

        int id = ois.readInt();

        if (id == -1) {
            return null;
        } else {
            int connectionsSize = ois.readInt();

            MutableIntList[] connections = new MutableIntList[connectionsSize];

            for (int i = 0; i < connectionsSize; i++) {
                int levelM = i == 0 ? maxM0 : maxM;
                connections[i] = readIntArrayList(ois, levelM);
            }

            TItem item = itemSerializer.read(ois);

            boolean deleted = ois.readBoolean();

            return new Node<>(id, connections, item, deleted);
        }
    }

    private static <TItem> AtomicReferenceArray<Node<TItem>> readNodesArray(ObjectInputStream ois,
                                                                            ObjectSerializer<TItem> itemSerializer,
                                                                            int maxM0,
                                                                            int maxM)
            throws IOException, ClassNotFoundException {

        int size = ois.readInt();
        AtomicReferenceArray<Node<TItem>> nodes = new AtomicReferenceArray<>(size);

        for (int i = 0; i < nodes.length(); i++) {
            nodes.set(i, readNode(ois, itemSerializer, maxM0, maxM));
        }

        return nodes;
    }

    private static <TId> MutableObjectIntMap<TId> readMutableObjectIntMap(ObjectInputStream ois,
                                                                          ObjectSerializer<TId> itemIdSerializer)
            throws IOException, ClassNotFoundException {

        int size = ois.readInt();

        MutableObjectIntMap<TId> map = new ObjectIntHashMap<>(size);

        for (int i = 0; i < size; i++) {
            TId key = itemIdSerializer.read(ois);
            int value = ois.readInt();

            map.put(key, value);
        }
        return map;
    }

    private static <TId> MutableObjectLongMap<TId> readMutableObjectLongMap(ObjectInputStream ois,
                                                                            ObjectSerializer<TId> itemIdSerializer)
            throws IOException, ClassNotFoundException {

        int size = ois.readInt();

        MutableObjectLongMap<TId> map = new ObjectLongHashMap<>(size);

        for (int i = 0; i < size; i++) {
            TId key = itemIdSerializer.read(ois);
            long value = ois.readLong();

            map.put(key, value);
        }
        return map;
    }

    /**
     * Start the process of building a new HNSW index.
     *
     * @param dimensions the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param maxItemCount maximum number of items the index can hold
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance extends Comparable<TDistance>> Builder<TVector, TDistance> newBuilder(
            int dimensions,
            DistanceFunction<TVector, TDistance> distanceFunction,
            int maxItemCount) {

        Comparator<TDistance> distanceComparator = Comparator.naturalOrder();

        return new Builder<>(false, dimensions, distanceFunction, distanceComparator, maxItemCount);
    }

    /**
     * Creates an immutable empty index.
     *
     * @return the empty index
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> empty() {
        Builder<TVector, TDistance> builder = new Builder<>(true, 0, new DistanceFunction<TVector, TDistance>() {
            @Override
            public TDistance distance(TVector u, TVector v) {
                throw new UnsupportedOperationException();
            }
        }, new DummyComparator<>(), 0);
        return builder.build();
    }

    /**
     * Start the process of building a new HNSW index.
     *
     * @param dimensions the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param distanceComparator used to compare distances
     * @param maxItemCount maximum number of items the index can hold
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance> Builder<TVector, TDistance> newBuilder(
            int dimensions,
            DistanceFunction<TVector, TDistance> distanceFunction,
            Comparator<TDistance> distanceComparator,
            int maxItemCount) {

        return new Builder<>(false, dimensions, distanceFunction, distanceComparator, maxItemCount);
    }

    private int assignLevel(TId value, double lambda) {

        // by relying on the external id to come up with the level, the graph construction should be a lot mor stable
        // see : https://github.com/nmslib/hnswlib/issues/28

        int hashCode = value.hashCode();

        byte[] bytes = new byte[]{
                (byte) (hashCode >> 24),
                (byte) (hashCode >> 16),
                (byte) (hashCode >> 8),
                (byte) hashCode
        };

        double random = Math.abs((double) Murmur3.hash32(bytes) / (double) Integer.MAX_VALUE);

        double r = -Math.log(random) * lambda;
        return (int) r;
    }

    /**
     * 判断 x是否小于y
     * @param x
     * @param y
     * @return
     */
    private boolean lt(TDistance x, TDistance y) {
        return maxValueDistanceComparator.compare(x, y) < 0;
    }

    private boolean gt(TDistance x, TDistance y) {
        return maxValueDistanceComparator.compare(x, y) > 0;
    }

    class ExactView implements Index<TId, TVector, TItem, TDistance> {

        private static final long serialVersionUID = 1L;

        @Override
        public int size() {
            return HnswIndex.this.size();
        }

        @Override
        public Optional<TItem> get(TId tId) {
            return HnswIndex.this.get(tId);
        }


        @Override
        public Collection<TItem> items() {
            return HnswIndex.this.items();
        }

        @Override
        public List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k) {

            Comparator<SearchResult<TItem, TDistance>> comparator = Comparator
                    .<SearchResult<TItem, TDistance>>naturalOrder()
                    .reversed();

            PriorityQueue<SearchResult<TItem, TDistance>> queue = new PriorityQueue<>(k, comparator);

            for (int i = 0; i < nodeCount; i++) {
                Node<TItem> node = nodes.get(i);
                if (node == null || node.deleted) {
                    continue;
                }

                TDistance distance = distanceFunction.distance(node.item.vector(), vector);

                SearchResult<TItem, TDistance> searchResult = new SearchResult<>(node.item, distance, maxValueDistanceComparator);
                queue.add(searchResult);

                if (queue.size() > k) {
                    queue.poll();
                }
            }

            List<SearchResult<TItem, TDistance>> results = new ArrayList<>(queue.size());

            SearchResult<TItem, TDistance> result;
            while ((result = queue.poll()) != null) { // if you iterate over a priority queue the order is not guaranteed
                results.add(0, result);
            }

            return results;
        }

        @Override
        public boolean add(TItem item) {
            return HnswIndex.this.add(item);
        }

        @Override
        public boolean remove(TId id, long version) {
            return HnswIndex.this.remove(id, version);
        }

        @Override
        public void save(OutputStream out) throws IOException {
            HnswIndex.this.save(out);
        }

        @Override
        public void save(File file) throws IOException {
            HnswIndex.this.save(file);
        }

        @Override
        public void save(Path path) throws IOException {
            HnswIndex.this.save(path);
        }

        @Override
        public void addAll(Collection<TItem> items) throws InterruptedException {
            HnswIndex.this.addAll(items);
        }

        @Override
        public void addAll(Collection<TItem> items, ProgressListener listener) throws InterruptedException {
            HnswIndex.this.addAll(items, listener);
        }

        @Override
        public void addAll(Collection<TItem> items, int numThreads, ProgressListener listener, int progressUpdateInterval) throws InterruptedException {
            HnswIndex.this.addAll(items, numThreads, listener, progressUpdateInterval);
        }
    }

    class ItemIterator implements Iterator<TItem> {

        private int done = 0;
        private int index = 0;

        @Override
        public boolean hasNext() {
            return done < HnswIndex.this.size();
        }

        @Override
        public TItem next() {
            Node<TItem> node;

            do {
                node = HnswIndex.this.nodes.get(index++);
            } while(node == null || node.deleted);

            done++;

            return node.item;
        }
    }

    static class Node<TItem> implements Serializable {

        private static final long serialVersionUID = 1L;

        final int id;

        final MutableIntList[] connections; // MutableIntList，int列表，int相比Integer性能更好

        volatile TItem item;

        volatile boolean deleted;

        Node(int id, MutableIntList[] connections, TItem item, boolean deleted) {
            this.id = id;
            this.connections = connections;
            this.item = item;
            this.deleted = deleted;
        }

        int maxLevel() {
            return this.connections.length - 1;
        }
    }

    static class NodeIdAndDistance<TDistance> implements Comparable<NodeIdAndDistance<TDistance>> {

        final int nodeId;
        final TDistance distance;
        final Comparator<TDistance> distanceComparator;

        NodeIdAndDistance(int nodeId, TDistance distance, Comparator<TDistance> distanceComparator) {
            this.nodeId = nodeId;
            this.distance = distance;
            this.distanceComparator = distanceComparator;
        }

        @Override
        public int compareTo(NodeIdAndDistance<TDistance> o) {
            return distanceComparator.compare(distance, o.distance);
        }
    }


    static class MaxValueComparator<TDistance> implements Comparator<TDistance>, Serializable  {

        private static final long serialVersionUID = 1L;

        private final Comparator<TDistance> delegate;

        MaxValueComparator(Comparator<TDistance> delegate) {
            this.delegate = delegate;
        }

        @Override
        public int compare(TDistance o1, TDistance o2) {
            return o1 == null ? o2 == null ? 0 : 1
                    : o2 == null ? -1 : delegate.compare(o1, o2);
        }

        static <TDistance> TDistance maxValue() {
            return null;
        }
    }

    /**
     * Base class for HNSW index builders.
     *
     * @param <TBuilder> Concrete class that extends from this builder
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of items stored in the index
     */
    public static abstract class BuilderBase<TBuilder extends BuilderBase<TBuilder, TVector, TDistance>, TVector, TDistance> {

        public static final int DEFAULT_M = 10;
        public static final int DEFAULT_EF = 10;
        public static final int DEFAULT_EF_CONSTRUCTION = 200;
        public static final boolean DEFAULT_REMOVE_ENABLED = false;

        boolean immutable;
        int dimensions;
        DistanceFunction<TVector, TDistance> distanceFunction;
        Comparator<TDistance> distanceComparator;

        int maxItemCount;

        int m = DEFAULT_M;
        int ef = DEFAULT_EF;
        int efConstruction = DEFAULT_EF_CONSTRUCTION;
        boolean removeEnabled = DEFAULT_REMOVE_ENABLED;

        BuilderBase(boolean immutable,
                    int dimensions,
                    DistanceFunction<TVector, TDistance> distanceFunction,
                    Comparator<TDistance> distanceComparator,
                    int maxItemCount) {
            this.immutable = immutable;
            this.dimensions = dimensions;
            this.distanceFunction = distanceFunction;
            this.distanceComparator = distanceComparator;
            this.maxItemCount = maxItemCount;
        }

        abstract TBuilder self();

        /**
         * Sets the number of bi-directional links created for every new element during construction. Reasonable range
         * for m is 2-100. Higher m work better on datasets with high intrinsic dimensionality and/or high recall,
         * while low m work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter
         * also determines the algorithm's memory consumption.
         * As an example for d = 4 random vectors optimal m for search is somewhere around 6, while for high dimensional
         * datasets (word embeddings, good face descriptors), higher M are required (e.g. m = 48, 64) for optimal
         * performance at high recall. The range mM = 12-48 is ok for the most of the use cases. When m is changed one
         * has to update the other parameters. Nonetheless, ef and efConstruction parameters can be roughly estimated by
         * assuming that m  efConstruction is a constant.
         *
         * @param m the number of bi-directional links created for every new element during construction
         * @return the builder.
         */
        public TBuilder withM(int m) {
            this.m = m;
            return self();
        }

        /**
         * `
         * The parameter has the same meaning as ef, but controls the index time / index precision. Bigger efConstruction
         * leads to longer construction, but better index quality. At some point, increasing efConstruction does not
         * improve the quality of the index. One way to check if the selection of ef_construction was ok is to measure
         * a recall for M nearest neighbor search when ef = efConstruction: if the recall is lower than 0.9, then
         * there is room for improvement.
         *
         * @param efConstruction controls the index time / index precision
         * @return the builder
         */
        public TBuilder withEfConstruction(int efConstruction) {
            this.efConstruction = efConstruction;
            return self();
        }

        /**
         * The size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more
         * accurate but slower search. The value ef of can be anything between k and the size of the dataset.
         *
         * @param ef size of the dynamic list for the nearest neighbors
         * @return the builder
         */
        public TBuilder withEf(int ef) {
            this.ef = ef;
            return self();
        }

        /**
         * Call to enable support for the experimental remove operation. Indices that support removes will consume more
         * memory.
         *
         * @return the builder
         */
        public TBuilder withRemoveEnabled() {
            this.removeEnabled = true;
            return self();
        }
    }


    /**
     * Builder for initializing an {@link HnswIndex} instance.
     *
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static class Builder<TVector, TDistance> extends BuilderBase<Builder<TVector, TDistance>, TVector, TDistance> {

        /**
         * Constructs a new {@link Builder} instance.
         *
         * @param dimensions the dimensionality of the vectors stored in the index
         * @param distanceFunction the distance function
         * @param maxItemCount     the maximum number of elements in the index
         */
        Builder(boolean immutable,
                int dimensions,
                DistanceFunction<TVector, TDistance> distanceFunction,
                Comparator<TDistance> distanceComparator,
                int maxItemCount) {

            super(immutable, dimensions, distanceFunction, distanceComparator, maxItemCount);
        }

        @Override
        Builder<TVector, TDistance> self() {
            return this;
        }

        /**
         * Register the serializers used when saving the index.
         *
         * @param itemIdSerializer serializes the key of the item
         * @param itemSerializer   serializes the
         * @param <TId>            Type of the external identifier of an item
         * @param <TItem>          implementation of the Item interface
         * @return the builder
         */
        public <TId, TItem extends Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, TDistance> withCustomSerializers(ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {
            return new RefinedBuilder<>(immutable, dimensions, distanceFunction, distanceComparator, maxItemCount, m, ef, efConstruction,
                    removeEnabled, itemIdSerializer, itemSerializer);
        }

        /**
         * Build the index that uses java object serializers to store the items when reading and writing the index.
         *
         * @param <TId>   Type of the external identifier of an item
         * @param <TItem> implementation of the Item interface
         * @return the hnsw index instance
         */
        public <TId, TItem extends Item<TId, TVector>> HnswIndex<TId, TVector, TItem, TDistance> build() {
            ObjectSerializer<TId> itemIdSerializer = new JavaObjectSerializer<>();
            ObjectSerializer<TItem> itemSerializer = new JavaObjectSerializer<>();

            return withCustomSerializers(itemIdSerializer, itemSerializer)
                    .build();
        }

    }

    /**
     * Extension of {@link Builder} that has knows what type of item is going to be stored in the index.
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static class RefinedBuilder<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
            extends BuilderBase<RefinedBuilder<TId, TVector, TItem, TDistance>, TVector, TDistance> {

        private ObjectSerializer<TId> itemIdSerializer;
        private ObjectSerializer<TItem> itemSerializer;

        RefinedBuilder(boolean immutable,
                       int dimensions,
                       DistanceFunction<TVector, TDistance> distanceFunction,
                       Comparator<TDistance> distanceComparator,
                       int maxItemCount,
                       int m,
                       int ef,
                       int efConstruction,
                       boolean removeEnabled,
                       ObjectSerializer<TId> itemIdSerializer,
                       ObjectSerializer<TItem> itemSerializer) {

            super(immutable, dimensions, distanceFunction, distanceComparator, maxItemCount);

            this.m = m;
            this.ef = ef;
            this.efConstruction = efConstruction;
            this.removeEnabled = removeEnabled;

            this.itemIdSerializer = itemIdSerializer;
            this.itemSerializer = itemSerializer;
        }

        @Override
        RefinedBuilder<TId, TVector, TItem, TDistance> self() {
            return this;
        }

        /**
         * Register the serializers used when saving the index.
         *
         * @param itemIdSerializer serializes the key of the item
         * @param itemSerializer   serializes the
         * @return the builder
         */
        public RefinedBuilder<TId, TVector, TItem, TDistance> withCustomSerializers(
                ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {

            this.itemIdSerializer = itemIdSerializer;
            this.itemSerializer = itemSerializer;

            return this;
        }

        /**
         * Build the index.
         *
         * @return the hnsw index instance
         */
        public HnswIndex<TId, TVector, TItem, TDistance> build() {
            return new HnswIndex<>(this);
        }

    }

}

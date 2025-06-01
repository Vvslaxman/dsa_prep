# **Graph Algorithms Study Plan (4 Days)**



### **Day 1: Graph Basics, BFS, DFS, and Shortest Path Algorithms**

#### **Topics to Cover:**

1. **Introduction to Graphs**:
   * Definitions: Vertex, Edge, Directed vs Undirected, Weighted vs Unweighted.
   * **Graph Representation**: Adjacency Matrix vs Adjacency List.

2. **Graph Traversals**:
   * **Breadth-First Search (BFS)**: Implementation, use cases (level order traversal, shortest path in unweighted graphs).
   * **Depth-First Search (DFS)**: Implementation, use cases (topological sorting, cycle detection).

3. **Shortest Path Algorithms**:
   * **Dijkstra’s Algorithm**: Shortest path for weighted graphs (positive weights).
   * **Bellman-Ford Algorithm**: Shortest path with negative weights (detect negative cycles).
   * **Floyd-Warshall Algorithm**: All-pairs shortest path (important for smaller graphs).

4. **Basic Problems**:
   * BFS and DFS on unweighted/weighted graphs.
   * Find the shortest path in unweighted graphs (BFS).
   * Detect if a graph is bipartite.

#### **Approximate Time Allocation**:
* **Concepts + Code Explanation**: 3-4 hours
* **Practice Problems**: 3-4 hours


### **Day 2: Minimum Spanning Tree (MST), Topological Sort, SCC, and Cycle Detection**

#### **Topics to Cover:**

1. **Minimum Spanning Tree**:
   * **Prim’s Algorithm**: Greedy approach to find MST.
   * **Kruskal’s Algorithm**: Another approach for MST using union-find (disjoint set).

2. **Topological Sorting**:
   * **Topological Sort**: For Directed Acyclic Graphs (DAGs). Use DFS or Kahn’s algorithm.

3. **Cycle Detection**:
   * **DFS Cycle Detection**: Detect cycles in directed and undirected graphs.
   * **Union-Find Algorithm**: Detect cycles in an undirected graph using disjoint-set (important for Kruskal’s algorithm).

4. **Strongly Connected Components (SCC)**:
   * **Kosaraju’s Algorithm**: Finding SCC in directed graphs.
   * **Tarjan’s Algorithm**: Another method for SCC using DFS.

5. **Basic Problems**:
   * Find the MST using Prim’s and Kruskal’s.
   * Topological Sort on a given DAG.
   * Detect cycle in an undirected/directed graph.
   * Find SCCs using Kosaraju/Tarjan.

#### **Approximate Time Allocation**:
* **Concepts + Code Explanation**: 3-4 hours
* **Practice Problems**: 3-4 hours


### **Day 3: Advanced Graph Algorithms (Flow, Matching, Dynamic Programming)**

#### **Topics to Cover:**

1. **Max Flow and Min Cut**:
   * **Ford-Fulkerson Algorithm**: Max flow using augmenting paths (Edmonds-Karp for implementation).
   * **Edmonds-Karp**: BFS-based solution for max flow.
   * **Min-Cut Theorem**: Relationship between max flow and min cut.

2. **Graph Coloring**:
   * **Graph Coloring Problem**: Assigning colors to nodes such that no two adjacent nodes share the same color.
   * **Greedy Coloring**: Efficient way to color nodes.

3. **Dynamic Programming on Graphs**:
   * **Longest Path** (for DAGs).
   * **Bellman-Ford** for shortest paths with negative weights (as part of dynamic programming).
   * **Traveling Salesman Problem (TSP)**: Naive solution using DP and backtracking.

4. **Advanced Problems**:
   * Implement Max Flow/Min Cut algorithms.
   * Solve TSP using Dynamic Programming on a graph.
   * Graph coloring problems (e.g., 3-coloring).

#### **Approximate Time Allocation**:
* **Concepts + Code Explanation**: 4-5 hours
* **Practice Problems**: 2-3 hours



### **Day 4: Graph Applications, Advanced Topics, and Review**

#### **Topics to Cover:**

1. **Applications of Graphs**:
   * **Shortest Path in a Grid** (e.g., A* Algorithm).
   * **Network Flow Problems** (like bipartite matching).
   * **Route Planning in Maps**: Using BFS/DFS/Dijkstra for finding paths in a graph.

2. **Advanced Topics**:
   * **Eulerian Path and Circuit**: Conditions for the existence of Eulerian paths and circuits.
   * **Hamiltonian Path**: Conditions for Hamiltonian paths (NP-complete problem).
   * **2-Coloring/Planarity**: Checking if a graph is planar and 2-colorable.

3. **Problem Solving Session**:
   * Solve complex graph problems from LeetCode/Codeforces/HackerRank that require multiple algorithms.
   * Practice coding on a whiteboard or in a timed environment to simulate real interview conditions.

4. **Review and Mock Interviews**:
   * Revise all topics.
   * Run through mock interview questions with a timer to simulate real interview conditions.

#### **Approximate Time Allocation**:
* **Concepts + Code Explanation**: 3-4 hours
* **Practice Problems**: 3-4 hours






# ✅ **Day 1: Graph Basics, BFS, DFS, and Shortest Path Algorithms**

## **1. Introduction to Graphs**

### ➤ **Basic Terminology**

* **Graph**: A collection of **nodes (vertices)** and **edges (connections)** between them.
* **Directed Graph**: Edges have direction (A ➝ B).
* **Undirected Graph**: Edges go both ways (A — B).
* **Weighted Graph**: Edges have weights/costs.
* **Unweighted Graph**: All edges are considered equal in weight (often used in BFS).

### ➤ **Graph Representations**

#### 🟢 **Adjacency List**

* A list where each index represents a node and stores its neighbors.
* **Space Efficient**: O(V + E)
* **Example**:

  ```python
  graph = {
      0: [1, 2],
      1: [0, 3],
      2: [0],
      3: [1]
  }
  ```

#### 🔵 **Adjacency Matrix**

* A 2D matrix where matrix\[i]\[j] = 1 (or weight) if there's an edge from i to j.
* **Space Complexity**: O(V²)
* **Faster edge checks**, but consumes more space.



## **2. Graph Traversals**

### ➤ **Breadth-First Search (BFS)**

* Uses a **queue**.
* Traverses **level-by-level**.
* Used for:

  * Finding **shortest path in unweighted graphs**
  * **Bipartite check**
  * **Connected components** in undirected graphs

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```


### ➤ **Depth-First Search (DFS)**

* Uses **recursion** or **stack**.
* Explores **as far as possible** along one branch before backtracking.
* Used for:

  * **Cycle detection**
  * **Topological sort**
  * **Connected components**

```python
def dfs(graph, node, visited):
    visited.add(node)
    print(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```



## **3. Shortest Path Algorithms**

### ➤ **Dijkstra's Algorithm**

* **Greedy approach**.
* For **weighted graphs** with **non-negative weights**.
* Uses a **priority queue (min-heap)**.

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    min_heap = [(0, start)]
    
    while min_heap:
        dist, node = heapq.heappop(min_heap)
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(min_heap, (new_dist, neighbor))
    return distances
```


### ➤ **Bellman-Ford Algorithm**

* Handles **negative weights**.
* Detects **negative weight cycles**.
* Time complexity: O(VE)

```python
def bellman_ford(edges, V, source):
    dist = [float('inf')] * V
    dist[source] = 0

    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return "Negative Cycle Detected"
    return dist
```


### ➤ **Floyd-Warshall Algorithm**

* All-pairs shortest path.
* Works with negative weights (no negative cycles).
* Time complexity: O(V³)

```python
def floyd_warshall(matrix):
    V = len(matrix)
    for k in range(V):
        for i in range(V):
            for j in range(V):
                matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])
```



## **4. Practice Problems**

* **BFS**: Shortest path in unweighted graph.
* **DFS**: Connected components, cycle detection.
* **Bipartite Check**: BFS/DFS with coloring.
* **Shortest Paths**: Implement Dijkstra, Bellman-Ford.



# ✅ **Day 2: MST, Topological Sort, Cycle Detection, SCC**



## **1. Minimum Spanning Tree (MST)**

### ➤ **Prim’s Algorithm**

* Start from any node.
* Use a **min-heap** to pick the smallest edge.
* Time: O(E log V) with priority queue.

```python
import heapq

def prim(graph, start):
    visited = set()
    min_heap = [(0, start)]
    total_cost = 0
    
    while min_heap:
        cost, node = heapq.heappop(min_heap)
        if node not in visited:
            visited.add(node)
            total_cost += cost
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(min_heap, (weight, neighbor))
    return total_cost
```



### ➤ **Kruskal’s Algorithm**

* Sort edges by weight.
* Use **Union-Find** to avoid cycles.

```python
def kruskal(edges, V):
    parent = list(range(V))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    edges.sort(key=lambda x: x[2])
    mst_cost = 0

    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            mst_cost += w
    return mst_cost
```



## **2. Topological Sort (DAGs Only)**

### ➤ **DFS-Based Approach**

* Perform DFS and push nodes to a stack after visiting all neighbors.

```python
def topo_sort_dfs(graph):
    visited = set()
    stack = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)
    return stack[::-1]
```


### ➤ **Kahn’s Algorithm (BFS-based)**

* Use in-degree array to track entry count of nodes.

```python
from collections import deque, defaultdict

def kahn_topological_sort(V, edges):
    graph = defaultdict(list)
    in_degree = [0] * V

    # Build the graph and compute in-degrees
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # Queue for all vertices with in-degree 0
    queue = deque([i for i in range(V) if in_degree[i] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) != V:
        return "Cycle detected - No Topological Ordering exists"
    
    return topo_order
```


## **3. Cycle Detection**

### ➤ **Undirected Graph (DFS)**

```python
def detect_cycle_undirected(graph, node, visited, parent):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            if detect_cycle_undirected(graph, neighbor, visited, node):
                return True
        elif neighbor != parent:
            return True
    return False
```

### ➤ **Directed Graph (DFS + Recursion Stack)**

```python
def detect_cycle_directed(graph):
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited and dfs(neighbor):
                return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False
```



## **4. Strongly Connected Components (SCC)**

### ➤ **Kosaraju’s Algorithm**

1. Perform **DFS** and fill nodes by **finish time**.
2. **Transpose the graph**.
3. Do **DFS** in reverse finish time order on the transposed graph.

### ➤ **Tarjan’s Algorithm**

* Use **low-link values** and **stack**.
* More efficient: O(V + E).


## **5. Practice Problems**

* **MST**: Find using Prim’s/Kruskal’s.
* **Topo Sort**: Validate course scheduling.
* **Cycle Detection**: Directed/Undirected.
* **SCC**: Kosaraju’s algorithm on directed graphs.


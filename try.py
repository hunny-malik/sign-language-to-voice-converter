class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size
        self.size = [1] * size  # Track the size of each component

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
                self.size[root_u] += self.size[root_v]
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
                self.size[root_v] += self.size[root_u]
            else:
                self.parent[root_v] = root_u
                self.size[root_u] += self.size[root_v]
                self.rank[root_u] += 1

def num_single_cells_remaining(M, N, grid):
    def cell_index(r, c):
        return r * N + c
    
    uf = UnionFind(M * N)
    
    for r in range(M):
        for c in range(N):
            if grid[r][c] == 'O':
                if c + 1 < N and grid[r][c + 1] == 'O':
                    uf.union(cell_index(r, c), cell_index(r, c + 1))
                if r + 1 < M and grid[r + 1][c] == 'O':
                    uf.union(cell_index(r, c), cell_index(r + 1, c))
    
    remaining_single_cells = 0
    for r in range(M):
        for c in range(N):
            if grid[r][c] == 'X':
                remaining_single_cells += 1  # Restricted cells are always single
            else:
                if uf.size[uf.find(cell_index(r, c))] == 1:
                    remaining_single_cells += 1  # Uncoupled empty cells

    return remaining_single_cells

if __name__ == "__main__":
    M, N = map(int, input().split())
    grid = [input().strip() for _ in range(M)]
    
    print(num_single_cells_remaining(M, N, grid))

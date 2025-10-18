n, m = map(int, input().split())
base_pattern = [input() for _ in range(n)]
k = int(input())
stones = set()
for _ in range(k):
    x, y = map(int, input().split())
    stones.add((x, y))

base_fragments = []
for r in range(n):
    for c in range(m):
        if base_pattern[r][c] == '1':
            base_fragments.append((r, c))

count = 0
for stone_x, stone_y in stones:
    is_match = True
    for frag_r, frag_c in base_fragments:
        abs_x = stone_x + frag_r
        abs_y = stone_y + frag_c
        if (abs_x, abs_y) not in stones:
            is_match = False
            break
    if is_match:
        count += 1

print(count)

n, m = map(int, input().split())
base_pattern = [input() for _ in range(n)]
k = int(input())
stones = set()
for _ in range(k):
    x, y = map(int, input().split())
    stones.add((x, y))

base_fragments = []
for r in range(n):
    for c in range(m):
        if base_pattern[r][c] == '1':
            base_fragments.append((r, c))

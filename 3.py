a, b, c = map(int, input().split())

r1, k1 = map(int, input().split())
r2, k2 = map(int, input().split())
r3, k3 = map(int, input().split())

def calculate_packages(count):
    ml_needed = count * 400
    return (ml_needed + 999) // 1000

packages1 = calculate_packages(a)
packages2 = calculate_packages(b)
packages3 = calculate_packages(c)

print(packages1, packages2, packages3)

total_kopecks = (r1 * 100 + k1) * packages1 + (r2 * 100 + k2) * packages2 + (r3 * 100 + k3) * packages3

total_rubles = total_kopecks // 100
remaining_kopecks = total_kopecks % 100

print(total_rubles, remaining_kopecks)

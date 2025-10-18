def solve():
    n = int(input())
    s = input().strip()

    has_dash = '-' in s
    #-------------------от A (a) до Y (y)
    #-------------------abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-
    alowed_chars = set("abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXY-")
    if not all(c in alowed_chars for c in s):
        print("Error!")
        return

    if has_dash:
        if s[0] == '-' or s[-1] == '-' or '--' in s:
            print("Error!")
            return
        for ch in s:
            if ch != '-' and ch.islower():
                print("Error!")
                return
        words = s.split('-')
        result = words[0].lower()
        for word in words[1:]:
            result += word.capitalize()
        print(result)
    else:
        if s[0].isupper():
            print("Error!")

            # print('sycle')
            return
        for i in range(1, len(s)):
            if s[i].isupper() and s[i-1].isupper():
                print("Error!")
                return
        words = []
        current_word = s[0]
        for i in range(1, len(s)):
            if s[i].isupper():
                words.append(current_word.upper())
                current_word = s[i].lower()
            else:
                current_word += s[i]
        words.append(current_word.upper())
        print('-'.join(words))

if __name__ == "__main__":
    solve()

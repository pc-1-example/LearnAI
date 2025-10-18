n = int(input().strip())
s = input().strip()

def is_first_form(s):
    if '-' in s:
        for char in s:
            if not (char == '-' or char.isupper()):
                return False
        if s[0] == '-' or s[-1] == '-':
            return False
        for i in range(1, len(s)):
            if s[i] == '-' and s[i-1] == '-':
                return False
        return True
    else:
        return all(char.isupper() for char in s)

def is_second_form(s):
    if '-' in s:
        return False
    if not s[0].islower():
        return False
    for i in range(1, len(s)):
        if s[i].isupper():
            if not s[i-1].islower():
                return False
    return True

if is_first_form(s):
    if '-' in s:
        words = s.split('-')
    else:
        words = [s]
    res = words[0].lower()
    for i in range(1, len(words)):
        word = words[i]
        res += word[0] + word[1:].lower()
    print(res)
elif is_second_form(s):
    words = []
    current = []
    for char in s:
        if char.isupper():
            if current:
                words.append(''.join(current))
                current = []
            current.append(char)
        else:
            current.append(char)
    if current:
        words.append(''.join(current))
    upper_words = [w.upper() for w in words]
    print('-'.join(upper_words))
else:
    print("Error!")

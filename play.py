
import re 

pred = "select sum ( c5_number ) from w where c1 in ( 'argus', 'james carruthers', 'hydrus' )"
print(pred)

pairs = re.finditer(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?\)', pred)
tokens = []
replacement = []
for idx, match in enumerate(pairs):
    start = match.start(0)
    end = match.end(0)
    col = pred[match.start(1):match.end(1)]
    ori1 = pred[match.start(2):match.end(2)]
    ori2 = pred[match.start(3):match.end(3)]
    if re.search(r"'\s*,\s*'", ori1+ori2): 
        print(ori1)
        print(ori2)
        print(re.search(r"'\s*,\s*'", ori1+ori2),'\n',ori1+ori2)
        continue
    to_replace = pred[start:end]

    token = str(idx) + '_'*(end-start-len(str(idx)))
    tokens.append(token)
    pred = pred[:start] + token + pred[end:]


print(f'C: part to be replaced: {to_replace}, col: {col}, string: {ori1}, {ori2}')
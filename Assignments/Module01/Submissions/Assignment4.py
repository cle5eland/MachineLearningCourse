from tabulate import tabulate

# vals = ['Call', 'to', 'call', 'or', 'FREE',
#      'claim', 'To', 'mobile', '&', 'Txt']
vals = ['to', 'you', 'I', 'a', 'the', 'and', 'is', 'in', 'i', 'u']

headers = ["words"]

table = []
for val in vals:
    table.append([val])

print(tabulate(table, headers, tablefmt="github"))

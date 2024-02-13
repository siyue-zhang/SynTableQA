

cols = ['1_id', '2_agg', '3_constituency', '4_constituency_number', '5_region', '6_name', '7_name_first', '8_name_second', '9_party', '10_last_elected', '11_last_elected_number']
cols = ['_'.join(col.split('_')[1:]) for col in cols[2:]]
print(cols)
original_cols = []
processed_cols = []

for i in range(len(cols)):
    if i==0:
        current = cols[i]
        current_ = 'c1'
        count = 1
        original_cols.append(current_)
    else:
        if current in cols[i]:
            processed_cols.append(cols[i].replace(current, current_))
        else:
            current = cols[i]
            count += 1
            current_ = f'c{count}'
            original_cols.append(current_)

print(original_cols, processed_cols)

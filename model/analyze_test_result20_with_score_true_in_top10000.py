import os

path = 'test_rank20_complete_true_in_top10000/'
#path = 'de1000_01_f_model21_complete_only_keywords_in_top1000/'
sql_win = 0
dl_win = 0
tie = 0
diff_sum = 0
count = 0
top1_sql = 0
top1_dl = 0
top1_avg = 0
top20_sql = 0
top20_dl = 0
top20_avg = 0
top100_sql = 0
top100_dl = 0
top100_avg = 0
top200_sql = 0
top200_dl = 0
top200_avg = 0
top300_sql = 0
top300_dl = 0
top300_avg = 0
top400_sql = 0
top400_dl = 0
top400_avg = 0
top500_sql = 0
top500_dl = 0
top500_avg = 0

for filename in os.listdir(path):
    #print(filename)
    f = open(path + filename, 'r')
    for line in f.readlines()[1:]:
        line = line.strip()
        #print(line)
        items = line.split('\t')
        sentence = items[0]
        sql_rank = items[1]
        dl_rank = items[2]
        avg_rank=(float(dl_rank)+float(sql_rank))/2.0
        if int(sql_rank) > int(dl_rank):
            dl_win += 1
        elif int(sql_rank) < int(dl_rank):
            sql_win += 1
        elif int(sql_rank) == int(dl_rank):
            tie += 1

        diff = int(sql_rank) - int(dl_rank)
        #print(diff)
        diff_sum += diff
        
        if int(sql_rank) <= 1:
            top1_sql += 1
        if int(sql_rank) <= 20:
            top20_sql += 1
        if int(sql_rank) <= 100:
            top100_sql += 1
        if int(sql_rank) <= 200:
            top200_sql += 1
        if int(sql_rank) <= 300:
            top300_sql += 1
        if int(sql_rank) <= 400:
            top400_sql += 1
        if int(sql_rank) <= 500:
            top500_sql += 1
        if int(dl_rank) <= 1:
            top1_dl += 1
        if int(dl_rank) <= 20:
            top20_dl += 1
        if int(dl_rank) <= 100:
            top100_dl += 1
        if int(dl_rank) <= 200:
            top200_dl += 1
        if int(dl_rank) <= 300:
            top300_dl += 1
        if int(dl_rank) <= 400:
            top400_dl += 1
        if int(dl_rank) <= 500:
            top500_dl += 1
        if int(avg_rank) <= 1:
            top1_avg += 1
        if int(avg_rank) <= 20:
            top20_avg += 1
        if int(avg_rank) <= 100:
            top100_avg += 1
        if int(avg_rank) <= 200:
            top200_avg += 1
        if int(avg_rank) <= 300:
            top300_avg += 1
        if int(avg_rank) <= 400:
            top400_avg += 1
        if int(avg_rank) <= 500:
            top500_avg += 1
        count += 1
    f.close()

print('-----------------------------------')
print(f'Total sentences: {count}')
print('=======================================')
print(f'sql win: {sql_win}')
print(f'dl win: {dl_win}')
print(f'Tie: {tie}')
print('-----------------------------------')
print(f'Difference: {diff_sum}')
print(f'Average rank improvement: {diff_sum / count}')
print('-----------------------------------')
print(f'Top1_sql: {top1_sql}')
print(f'Top1_dl: {top1_dl}')
print(f'Top1_avg: {top1_avg}')
print('-----------------------------------')
print(f'Top20_sql: {top20_sql}')
print(f'Top20_dl: {top20_dl}')
print(f'Top20_avg: {top20_avg}')
print('-----------------------------------')
print(f'Top100_sql: {top100_sql}')
print(f'Top100_dl: {top100_dl}')
print(f'Top100_avg: {top100_avg}')
print('-----------------------------------')
print(f'Top200_sql: {top200_sql}')
print(f'Top200_dl: {top200_dl}')
print(f'Top200_avg: {top200_avg}')
print('-----------------------------------')
print(f'Top300_sql: {top300_sql}')
print(f'Top300_dl: {top300_dl}')
print(f'Top300_avg: {top300_avg}')
print('-----------------------------------')
print(f'Top400_sql: {top400_sql}')
print(f'Top400_dl: {top400_dl}')
print(f'Top400_avg: {top400_avg}')
print('-----------------------------------')
print(f'Top500_sql: {top500_sql}')
print(f'Top500_dl: {top500_dl}')
print(f'Top500_avg: {top500_avg}')
print('-----------------------------------')

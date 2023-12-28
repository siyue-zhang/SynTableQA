
# nt-13417  203_37
normal_sample_0 = """
Q: how many tracks were produced by milanna miles and rick long? 
answer A : 0 
answer B : 2 
col : id | agg | # | #_number | title | producer(s) | producer(s)_list | producer(s)_length | performer(s) | length | length_number | length_min | length_sec row 1 : 1 | 0 | 1 | 1.0 | "ova here" | da beatminerz | da beatminerz | 1 | krs-one | 2:56 | 2 | 2 | 56 row 2 : 2 | 0 | 2 | 2.0 | "things is about to change" | creative minds combined | creative minds combined | 1 | krs-one | 2:34 | 2 | 2 | 34 row 3 : 3 | 0 | 3 | 3.0 | "splash" | a-sharp, pleasure king | a-sharp, pleasure king | 2 | krs-one | 2:24 | 2 | 2 | 24 row 4 : 4 | 0 | 4 | 4.0 | "my people" | alumni | alumni | 1 | krs-one | 0:19 | 0 | 0 | 19 row 5 : 5 | 0 | 5 | 5.0 | "kreditz" | milann miles, rick long | milann miles, rick long | 2 | krs-one | 2:02 | 2 | 2 | 2 row 6 : 6 | 0 | 6 | 6.0 | "i remember" | j-roc, mad lion | j-roc, mad lion | 2 | krs-one | 4:03 | 4 | 4 | 3 row 7 : 7 | 0 | 7 | 7.0 | "down the charts" | milann miles, rick long | milann miles, rick long | 2 | krs-one | 2:10 | 2 | 2 | 10 row 8 : 8 | 0 | 8 | 8.0 | "you really don't want it" | jim bean | jim bean | 1 | None | 3:29 | 3 | 3 | 29 row 9 : 9 | 0 | 9 | 9.0 | "womanology" | soul supreme | soul supreme | 1 | krs-one | 3:54 | 3 | 3 | 54 row 10 : 10 | 0 | 10 | 10.0 | "2nd kreditz" | None | | 0 | krs-one | 0:46 | 0 | 0 | 46 row 11 : 11 | 0 | 11 | 11.0 | "stop it" | krs-one, mad lion | krs-one, mad lion | 2 | None | 1:38 | 1 | 1 | 38 row 12 : 12 | 0 | 12 | 12.0 | "problemz" | vanguard | vanguard | 1 | krs-one | 3:15 | 3 | 3 | 15 row 13 : 13 | 0 | 13 | 13.0 | "believe it!" | soul supreme | soul supreme | 1 | krs-one | 3:59 | 3 | 3 | 59

A: This is a table of music tracks, which contains track title, producer, and length columns. \
Each row represents a track. There are two rows, 5 and 7, which has milanna miles and rick long \
under the producer(s) column. So, based on the given table, there are at least two tracks \
were produced by milanna miles and rick long. So, Answer B is more correct than Answer A.
"""

# nt-970    203_535
normal_sample_1 = """
Q: which countries had the same number of gold medals as belarus? 
answer A : 2 
answer B : France, Ethiopia 
col : id | agg | rank | rank_number | nation | gold | gold_number | silver | silver_number | bronze | bronze_number | total | total_number row 1 : 1 | 0 | 1 | 1.0 | united states | 8 | 8.0 | 7 | 7.0 | 1 | 1.0 | 16 | 16.0 row 2 : 2 | 0 | 2 | 2.0 | russia | 7 | 7.0 | 7 | 7.0 | 5 | 5.0 | 19 | 19.0 row 3 : 3 | 0 | 3 | 3.0 | france | 3 | 3.0 | 3 | 3.0 | 2 | 2.0 | 8 | 8.0 row 4 : 4 | 0 | 4 | 4.0 | ethiopia | 3 | 3.0 | 2 | 2.0 | 2 | 2.0 | 7 | 7.0 row 5 : 5 | 0 | 5 | 5.0 | belarus | 3 | 3.0 | 1 | 1.0 | 3 | 3.0 | 7 | 7.0 row 6 : 6 | 0 | 6 | 6.0 | sweden | 2 | 2.0 | 1 | 1.0 | 2 | 2.0 | 5 | 5.0 row 7 : 7 | 0 | 7 | 7.0 | kenya | 2 | 2.0 | 1 | 1.0 | 1 | 1.0 | 4 | 4.0 row 8 : 8 | 0 | 7= | 7.0 | south africa | 2 | 2.0 | 1 | 1.0 | 1 | 1.0 | 4 | 4.0 row 9 : 9 | 0 | 9 | 9.0 | morocco | 2 | 2.0 | 1 | 1.0 | 0 | 0.0 | 3 | 3.0 row 10 : 10 | 0 | 10 | 10.0 | greece | 1 | 1.0 | 1 | 1.0 | 2 | 2.0 | 4 | 4.0 row 11 : 11 | 0 | 11 | 11.0 | cuba | 1 | 1.0 | 1 | 1.0 | 0 | 0.0 | 2 | 2.0 row 12 : 12 | 0 | 12 | 12.0 | italy | 1 | 1.0 | 0 | 0.0 | 2 | 2.0 | 3 | 3.0 row 13 : 13 | 0 | 13 | 13.0 | canada | 1 | 1.0 | 0 | 0.0 | 1 | 1.0 | 2 | 2.0 row 14 : 14 | 0 | 14 | 14.0 | algeria | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 15 : 15 | 0 | 14= | 14.0 | australia | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 16 : 16 | 0 | 14= | 14.0 | dominican republic | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 17 : 17 | 0 | 14= | 14.0 | ecuador | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 18 : 18 | 0 | 14= | 14.0 | lithuania | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 19 : 19 | 0 | 14= | 14.0 | mexico | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 20 : 20 | 0 | 14= | 14.0 | mozambique | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 21 : 21 | 0 | 14= | 14.0 | poland | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 22 : 22 | 0 | 14= | 14.0 | qatar | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 23 : 23 | 0 | 14= | 14.0 | saint kitts and nevis | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 24 : 24 | 0 | 24 | 24.0 | jamaica | 0 | 0.0 | 4 | 4.0 | 2 | 2.

A: This is a table of records of various countries, which contains nation, number of gold, silver, \
and bronze, and total number columns. Based on row 5, Belarus has 3 gold medals under gold column. \
Based on row 4, Ethiopia has 3 gold medals under gold column. Based on row 3, France has 3 gold medals \
under gold column. Answer A is a number which is not the country asked by the question. So Answer B \
is more correct than Answer A.
"""

# nt-2474   204_469
normal_sample_2 = """
Q: who was the last coach to only have 1 loss? 
answer A : ed sabre 
answer B : Joe Sewell 
col : id | agg | coach | years | years_list | years_length | years_list_minimum_number | years_list_minimum_parsed | years_list_minimum_year | years_list_maximum_number | years_list_maximum_parsed | years_list_maximum_year | seasons | seasons_number | wins | wins_number | losses | losses_number | ties | ties_number | pct | pct_number row 1 : 1 | 0 | o. b. "rip" sanderson | 1908 | 1908 | 1 | 1908 | 1908 | 1908 | 1908 | 1908 | 1908 | 1 | 1.0 | 5 | 5.0 | 1 | 1.0 | 0 | 0.0 | 0.833 | 0.833 row 2 : 2 | 0 | ed sabre | 1920 | 1920 | 1 | 1920 | 1920 | 1920 | 1920 | 1920 | 1920 | 1 | 1.0 | 3 | 3.0 | 1 | 1.0 | 0 | 0.0 | 0.75 | 0.75 row 3 : 3 | 0 | mack erwin | 1957-60 | 1957-60 | 1 | 1957 | 1957 | 1957 | 1960 | 1960 | 1960 | 4 | 4.0 | 62 | 62.0 | 30 | 30.0 | 1 | 1.0 | 0.672 | 0.672 row 4 : 4 | 0 | chal port | 1965-91 | 1965-91 | 1 | 1965 | 1965 | 1965 | 1991 | 1991 | 1991 | 27 | 27.0 | 641 | 641.0 | 386 | 386.0 | 2 | 2.0 | 0.624 | 0.624 row 5 : 5 | 0 | fred jordan | 1992-pres | 1992-pres | 1 | 1992 | 1992 | 1992 | 1992 | 1992 | 1992 | 22 | 22.0 | 726 | 726.0 | 552 | 552.0 | 0 | 0.0 | 0.568 | 0.568 row 6 : 6 | 0 | john d. mcmillan | 1952-53 | 1952-53 | 1 | 1952 | 1952 | 1952 | 1953 | 1953 | 1953 | 2 | 2.0 | 14 | 14.0 | 15 | 15.0 | 0 | 0.0 | 0.483 | 0.483 row 7 : 7 | 0 | jim newsome | 1961-64 | 1961-64 | 1 | 1961 | 1961 | 1961 | 1964 | 1964 | 1964 | 4 | 4.0 | 37 | 37.0 | 43 | 43.0 | 0 | 0.0 | 0.463 | 0.463 row 8 : 8 | 0 | bunzy o'neal | 1948 | 1948 | 1 | 1948 | 1948 | 1948 | 1948 | 1948 | 1948 | 1 | 1.0 | 6 | 6.0 | 7 | 7.0 | 0 | 0.0 | 0.462 | 0.462 row 9 : 9 | 0 | george c. rogers | 1914-15, 1921-24 | 1914-15, 1921-24 | 2 | 1914, 1921 | 1914, 1921 | 1914, 1921 | 1915, 1924 | 1915, 1924 | 1915, 1924 | 6 | 6.0 | 26 | 26.0 | 33 | 33.0 | 1 | 1.0 | 0.441 | 0.441 row 10 : 10 | 0 | fred montsdeoca | 1954-56 | 1954-56 | 1 | 1954 | 1954 | 1954 | 1956 | 1956 | 1956 | 2 | 2.0 | 22 | 22.0 | 31 | 31.0 | 0 | 0.0 | 0.415 | 0.415 row 11 : 11 | 0 | george a. schmick | 1910 | 1910 | 1 | 1910 | 1910 | 1910 | 1910 | 1910 | 1910 | 1 | 1.0 | 4 | 4.0 | 6 | 6.0 | 0 | 0.0 | 0.4 | 0.4 row 12 : 12 | 0 | thomas s. mcmillan | 1916, 1919 | 1916, 1919 | 2 | 1916, 1919 | 1916, 1919 | 1916, 1919 | 1916, 1919 | 1916, 1919 | 1916, 1919 | 2 | 2.0 | 8 | 8.0 | 12 | 12.0 | 0 | 0.0 | 0.4 | 0.4 row 13 : 13 | 0 | harold manley | 1954 | 1954 | 1 | 1954 | 1954 | 1954 | 1954 | 1954 | 1954 | 1 | 1.0 | 5 | 5.0 | 9 | 9.0 | 0 | 0.0 | 0.357 | 0.357 row 14 : 14 | 0 | "barney

A: This is a table of game records, which contains coach, years, losses, ties, pct columns. \
The row 14 is not complete, thus only the part of the table is provided. The row 2 has \
ed sabre under the the coach column and 1 under the losses column. Based on the losses column, \
the last row has 1 is the row 2. Joe Sewell is not under the coach column. \
So Answer A is more correct than Answer B.
"""




# nt-11898	204_40
large_sample_0 = """
what was the last race of the season? 
answer A : portimao 
answer B : MX Open 
col : id | agg | round | round_number | round | country | circuit | date | date_number | date_parsed | date_year | date_month | date_day | pole_position | fastest_lap | winning_rider | winning_team | report row 1 : 1 | 0 | 1 | 1.0 | r1 | qatar | losail | 23 february | 23 | xxxx-02-23 | -1 | 2 | 23 | troy corser | noriyuki haga | troy bayliss | xerox ducati | report row 2 : 2 | 0 | 1 | 1.0 | r2 | qatar | losail | 23 february | 23 | xxxx-02-23 | -1 | 2 | 23 | troy corser | fonsi nieto | fonsi nieto | alstare suzuki | report row 3 : 3 | 0 | 2 | 2.0 | r1 | australia | phillip island | 2 march | 2 | xxxx-03-02 | -1 | 3 | 2 | troy bayliss | troy bayliss | troy bayliss | xerox ducati | report row 4 : 4 | 0 | 2 | 2.0 | r2 | australia | phillip island | 2 march | 2 | xxxx-03-02 | -1 | 3 | 2 | troy bayliss | max biaggi | troy bayliss | xerox ducati | report row 5 : 5 | 0 | 3 | 3.0 | r1 | spain | valencia | 6 april | 6 | xxxx-04-06 | -1 | 4 | 6 | max neukirchner | noriyuki haga | lorenzo lanzi | team r.g | report row 6 : 6 | 0 | 3 | 3.0 | r2 | spain | valencia | 6 april | 6 | xxxx-04-06 | -1 | 4 | 6 | max neukirchner | carlos checa | noriyuki haga | yamaha motor italia | report row 7 : 7 | 0 | 4 | 4.0 | r1 | netherlands | assen | 27 april | 27 | xxxx-04-27 | -1 | 4 | 27 | troy bayliss | max neukirchner | troy bayliss | xerox ducati | report row 8 : 8 | 0 | 4 | 4.0 | r2 | netherlands | assen | 27 april | 27 | xxxx-04-27 | -1 | 4 | 27 | troy bayliss | troy bayliss | troy bayliss | xerox ducati | report row 9 : 9 | 0 | 5 | 5.0 | r1 | italy | monza | 11 may | 11 | xxxx-05-11 | -1 | 5 | 11 | troy bayliss | noriyuki haga | max neukirchner | alstare suzuki | report row 10 : 10 | 0 | 5 | 5.0 | r2 | italy | monza | 11 may | 11 | xxxx-05-11 | -1 | 5 | 11 | troy bayliss | noriyuki haga | noriyuki haga | yamaha motor italia | report row 11 : 11 | 0 | 6 | 6.0 | r1 | united states | miller motorsports park | 1 june | 1 | xxxx-06-01 | -1 | 6 | 1 | carlos checa | carlos checa | carlos checa | ten kate honda | report row 12 : 12 | 0 | 6 | 6.0 | r2 | united states | miller motorsports park | 1 june | 1 | xxxx-06-01 | -1 | 6 | 1 | carlos checa | carlos checa | carlos checa | ten kate honda | report row 13 : 13 | 0 | 7 | 7.0 | r1 | germany | nurburgring | 15 june | 15 | xxxx-06-15 | -1 | 6 | 15 | max neukirchner | troy bayliss | noriyuki haga | yamaha motor italia | report row 14 : 14 | 0 | 7 | 7.0 | r2 | germany | nurburgring | 15 june | 15 | xxxx-06-15 | -1 | 6 | 15 |

A: This is a table of racing records of the season, which contains round, country, circuit, date, pole position, \
fastest lap, winning rider, winning team, and report columns. The row 14 is not complete, thus only \
the part of the table is provided. The last row in the provided incomplete table has nurburgring under the season column. \
There maybe more records in the original table. So the last race of the season can be something else other than nurburgring. \
Answer B MX Open is not under the season column in any row. So Answer A is more correct than Answer B.
"""




# nu-714    204_326
test_sample_0 ="""
Q: Name the format with the highest value in `` decimal 128 ''.
answer A : exponent range 
answer B : Emax = 3√ó2W-1
col : id | agg | decimal32 | decimal32_number | decimal64 | decimal64_number | decimal128 | decimal128_number | decimal(32k) | format row 1 : 1 | 0 | 1 | 1.0 | 1 | 1.0 | 1 | 1.0 | 1 | sign field (bits) row 2 : 2 | 0 | 5 | 5.0 | 5 | 5.0 | 5 | 5.0 | 5 | combination field (bits) row 3 : 3 | 0 | 6 | 6.0 | 8 | 8.0 | 12 | 12.0 | w = 2√ók + 4 | exponent continuation field (bits) row 4 : 4 | 0 | 20 | 20.0 | 50 | 50.0 | 110 | 110.0 | t = 30√ók-10 | coefficient continuation field (bits) row 5 : 5 | 0 | 32 | 32.0 | 64 | 64.0 | 128 | 128.0 | 32√ók | total size (bits) row 6 : 6 | 0 | 7 | 7.0 | 16 | 16.0 | 34 | 34.0 | p = 3√ót/10+1 = 9√ók-2 | coefficient size (decimal digits) row 7 : 7 | 0 | 192 | 192.0 | 768 | 768.0 | 12288 | 12288.0 | 3√ó2w = 48√ó4k | exponent range row 8 : 8 | 0 | 96 | 96.0 | 384 | 384.0 | 6144 | 6144.0 | emax = 3√ó2w-1 | largest value is 9.99...√ó10emax row 9 : 9 | 0 | -95 | -95.0 | -383 | -383.0 | -6143 | -6143.0 | emin = 1-emax | smallest normalized value is 1.00...√ó10emin row 10 : 10 | 0 | -101 | -101.0 | -398 | -398.0 | -6176 | -6176.0 | etiny = 2-p-emax | smallest non-zero value is 1√ó10etiny

A:
"""
# GPT4
# ChatCompletionMessage(content='This is a table comparing the qualities of various decimal formats. 
# Based on the table, the highest value in "decimal128" column is 12288, which corresponds to the "exponent range" 
# in the format row. So, Answer A is the correct one.', role='assistant', function_call=None, tool_calls=None)

# system_prompt = """
# You are an assistant helping users choose the correct answer to the question and the table provided. \
# Your should choose one from the answer A and B provided and explain the reasoning. \
# In the last sentence of your response, you should write either "A is more likely to be correct." or "B is more likely to be correct." \
# If there is not enough information in the table, you should respond "A is more likely to be correct.". \
# """

system_prompt = """
You will get a table, a question, and candidate answers. In the table, each cell is separated by "|". \
You should choose the more likely candidate based on the provided table information. \
The last sentence should be your choice "A", "B", or "C".
"""

# nt-13417 203_37
normal_sample_0 = """
Q: how many tracks were produced by milanna miles and rick long? 
answer A : 0 
answer B : 2 
 col : id | agg | # | #_number | title | producer(s) | producer(s)_list | producer(s)_length | performer(s) | length | length_number | length_min | length_sec row 1 : 1 | 0 | 1 | 1.0 | "ova here" | da beatminerz | da beatminerz | 1 | krs-one | 2:56 | 2 | 2 | 56 row 2 : 2 | 0 | 2 | 2.0 | "things is about to change" | creative minds combined | creative minds combined | 1 | krs-one | 2:34 | 2 | 2 | 34 row 3 : 3 | 0 | 3 | 3.0 | "splash" | a-sharp, pleasure king | a-sharp, pleasure king | 2 | krs-one | 2:24 | 2 | 2 | 24 row 4 : 4 | 0 | 4 | 4.0 | "my people" | alumni | alumni | 1 | krs-one | 0:19 | 0 | 0 | 19 row 5 : 5 | 0 | 5 | 5.0 | "kreditz" | milann miles, rick long | milann miles, rick long | 2 | krs-one | 2:02 | 2 | 2 | 2 row 6 : 6 | 0 | 6 | 6.0 | "i remember" | j-roc, mad lion | j-roc, mad lion | 2 | krs-one | 4:03 | 4 | 4 | 3 row 7 : 7 | 0 | 7 | 7.0 | "down the charts" | milann miles, rick long | milann miles, rick long | 2 | krs-one | 2:10 | 2 | 2 | 10 row 8 : 8 | 0 | 8 | 8.0 | "you really don't want it" | jim bean | jim bean | 1 | None | 3:29 | 3 | 3 | 29 row 9 : 9 | 0 | 9 | 9.0 | "womanology" | soul supreme | soul supreme | 1 | krs-one | 3:54 | 3 | 3 | 54 row 10 : 10 | 0 | 10 | 10.0 | "2nd kreditz" | None | | 0 | krs-one | 0:46 | 0 | 0 | 46 row 11 : 11 | 0 | 11 | 11.0 | "stop it" | krs-one, mad lion | krs-one, mad lion | 2 | None | 1:38 | 1 | 1 | 38 row 12 : 12 | 0 | 12 | 12.0 | "problemz" | vanguard | vanguard | 1 | krs-one | 3:15 | 3 | 3 | 15 row 13 : 13 | 0 | 13 | 13.0 | "believe it!" | soul supreme | soul supreme | 1 | krs-one | 3:59 | 3 | 3 | 59

A: This is a table of music tracks, which contains track title, producer, and length  columns. \
Each row represents a track. There are two rows, 5 and 7, which has milanna miles and rick long \
under the producer(s) column. So, based on the given table, there are at least two tracks \
were produced by milanna miles and rick long. So, Answer B is more correct than Answer A.
"""

# nt-970 203_535
normal_sample_1 = """
Q: which countries had the same number of gold medals as belarus? 
answer A : 2 
answer B : France, Ethiopia 
 col : id | agg | rank | rank_number | nation | gold | gold_number | silver | silver_number | bronze | bronze_number | total | total_number row 1 : 1 | 0 | 1 | 1.0 | united states | 8 | 8.0 | 7 | 7.0 | 1 | 1.0 | 16 | 16.0 row 2 : 2 | 0 | 2 | 2.0 | russia | 7 | 7.0 | 7 | 7.0 | 5 | 5.0 | 19 | 19.0 row 3 : 3 | 0 | 3 | 3.0 | france | 3 | 3.0 | 3 | 3.0 | 2 | 2.0 | 8 | 8.0 row 4 : 4 | 0 | 4 | 4.0 | ethiopia | 3 | 3.0 | 2 | 2.0 | 2 | 2.0 | 7 | 7.0 row 5 : 5 | 0 | 5 | 5.0 | belarus | 3 | 3.0 | 1 | 1.0 | 3 | 3.0 | 7 | 7.0 row 6 : 6 | 0 | 6 | 6.0 | sweden | 2 | 2.0 | 1 | 1.0 | 2 | 2.0 | 5 | 5.0 row 7 : 7 | 0 | 7 | 7.0 | kenya | 2 | 2.0 | 1 | 1.0 | 1 | 1.0 | 4 | 4.0 row 8 : 8 | 0 | 7= | 7.0 | south africa | 2 | 2.0 | 1 | 1.0 | 1 | 1.0 | 4 | 4.0 row 9 : 9 | 0 | 9 | 9.0 | morocco | 2 | 2.0 | 1 | 1.0 | 0 | 0.0 | 3 | 3.0 row 10 : 10 | 0 | 10 | 10.0 | greece | 1 | 1.0 | 1 | 1.0 | 2 | 2.0 | 4 | 4.0 row 11 : 11 | 0 | 11 | 11.0 | cuba | 1 | 1.0 | 1 | 1.0 | 0 | 0.0 | 2 | 2.0 row 12 : 12 | 0 | 12 | 12.0 | italy | 1 | 1.0 | 0 | 0.0 | 2 | 2.0 | 3 | 3.0 row 13 : 13 | 0 | 13 | 13.0 | canada | 1 | 1.0 | 0 | 0.0 | 1 | 1.0 | 2 | 2.0 row 14 : 14 | 0 | 14 | 14.0 | algeria | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 15 : 15 | 0 | 14= | 14.0 | australia | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 16 : 16 | 0 | 14= | 14.0 | dominican republic | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 17 : 17 | 0 | 14= | 14.0 | ecuador | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 18 : 18 | 0 | 14= | 14.0 | lithuania | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 19 : 19 | 0 | 14= | 14.0 | mexico | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 20 : 20 | 0 | 14= | 14.0 | mozambique | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 21 : 21 | 0 | 14= | 14.0 | poland | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 22 : 22 | 0 | 14= | 14.0 | qatar | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 23 : 23 | 0 | 14= | 14.0 | saint kitts and nevis | 1 | 1.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1.0 row 24 : 24 | 0 | 24 | 24.0 | jamaica | 0 | 0.0 | 4 | 4.0 | 2 | 2.

A: This is a table of records of various countries, which contains nation, number of gold, silver, \
and bronze, and total number  columns. Based on row 5, Belarus has 3 gold medals under gold  column. \
Based on row 4, Ethiopia has 3 gold medals under gold  column. Based on row 3, France has 3 gold medals \
under gold  column. Answer A is a number which is not the country asked by the question. \
So Answer B is more correct than Answer A.
"""

# nt-2474 204_469
normal_sample_2 = """
Q: who was the last coach to only have 1 loss? 
answer A : ed sabre 
answer B : Joe Sewell 
 col : id | agg | coach | years | years_list | years_length | years_list_minimum_number | years_list_minimum_parsed | years_list_minimum_year | years_list_maximum_number | years_list_maximum_parsed | years_list_maximum_year | seasons | seasons_number | wins | wins_number | losses | losses_number | ties | ties_number | pct | pct_number row 1 : 1 | 0 | o. b. "rip" sanderson | 1908 | 1908 | 1 | 1908 | 1908 | 1908 | 1908 | 1908 | 1908 | 1 | 1.0 | 5 | 5.0 | 1 | 1.0 | 0 | 0.0 | 0.833 | 0.833 row 2 : 2 | 0 | ed sabre | 1920 | 1920 | 1 | 1920 | 1920 | 1920 | 1920 | 1920 | 1920 | 1 | 1.0 | 3 | 3.0 | 1 | 1.0 | 0 | 0.0 | 0.75 | 0.75 row 3 : 3 | 0 | mack erwin | 1957-60 | 1957-60 | 1 | 1957 | 1957 | 1957 | 1960 | 1960 | 1960 | 4 | 4.0 | 62 | 62.0 | 30 | 30.0 | 1 | 1.0 | 0.672 | 0.672 row 4 : 4 | 0 | chal port | 1965-91 | 1965-91 | 1 | 1965 | 1965 | 1965 | 1991 | 1991 | 1991 | 27 | 27.0 | 641 | 641.0 | 386 | 386.0 | 2 | 2.0 | 0.624 | 0.624 row 5 : 5 | 0 | fred jordan | 1992-pres | 1992-pres | 1 | 1992 | 1992 | 1992 | 1992 | 1992 | 1992 | 22 | 22.0 | 726 | 726.0 | 552 | 552.0 | 0 | 0.0 | 0.568 | 0.568 row 6 : 6 | 0 | john d. mcmillan | 1952-53 | 1952-53 | 1 | 1952 | 1952 | 1952 | 1953 | 1953 | 1953 | 2 | 2.0 | 14 | 14.0 | 15 | 15.0 | 0 | 0.0 | 0.483 | 0.483 row 7 : 7 | 0 | jim newsome | 1961-64 | 1961-64 | 1 | 1961 | 1961 | 1961 | 1964 | 1964 | 1964 | 4 | 4.0 | 37 | 37.0 | 43 | 43.0 | 0 | 0.0 | 0.463 | 0.463 row 8 : 8 | 0 | bunzy o'neal | 1948 | 1948 | 1 | 1948 | 1948 | 1948 | 1948 | 1948 | 1948 | 1 | 1.0 | 6 | 6.0 | 7 | 7.0 | 0 | 0.0 | 0.462 | 0.462 row 9 : 9 | 0 | george c. rogers | 1914-15, 1921-24 | 1914-15, 1921-24 | 2 | 1914, 1921 | 1914, 1921 | 1914, 1921 | 1915, 1924 | 1915, 1924 | 1915, 1924 | 6 | 6.0 | 26 | 26.0 | 33 | 33.0 | 1 | 1.0 | 0.441 | 0.441 row 10 : 10 | 0 | fred montsdeoca | 1954-56 | 1954-56 | 1 | 1954 | 1954 | 1954 | 1956 | 1956 | 1956 | 2 | 2.0 | 22 | 22.0 | 31 | 31.0 | 0 | 0.0 | 0.415 | 0.415 row 11 : 11 | 0 | george a. schmick | 1910 | 1910 | 1 | 1910 | 1910 | 1910 | 1910 | 1910 | 1910 | 1 | 1.0 | 4 | 4.0 | 6 | 6.0 | 0 | 0.0 | 0.4 | 0.4 row 12 : 12 | 0 | thomas s. mcmillan | 1916, 1919 | 1916, 1919 | 2 | 1916, 1919 | 1916, 1919 | 1916, 1919 | 1916, 1919 | 1916, 1919 | 1916, 1919 | 2 | 2.0 | 8 | 8.0 | 12 | 12.0 | 0 | 0.0 | 0.4 | 0.4 row 13 : 13 | 0 | harold manley | 1954 | 1954 | 1 | 1954 | 1954 | 1954 | 1954 | 1954 | 1954 | 1 | 1.0 | 5 | 5.0 | 9 | 9.0 | 0 | 0.0 | 0.357 | 0.357 row 14 : 14 | 0 | "barney

A: This is a table of game records, which contains coach, years, losses, ties, pct  columns. \
The row 14 is not complete, thus only the part of the table is provided. The row 2 has \
ed sabre under the the coach  column and 1 under the losses  column. Based on the losses  column, \
the last row has 1 is the row 2. Joe Sewell is not under the coach  column. \
So Answer A is more correct than Answer B.
"""




# nt-11898 204_40
large_sample_0 = """
Q: what was the last race of the season? 
answer A : portimao 
answer B : MX Open 
 col : id | agg | round | round_number | round | country | circuit | date | date_number | date_parsed | date_year | date_month | date_day | pole_position | fastest_lap | winning_rider | winning_team | report row 1 : 1 | 0 | 1 | 1.0 | r1 | qatar | losail | 23 february | 23 | xxxx-02-23 | -1 | 2 | 23 | troy corser | noriyuki haga | troy bayliss | xerox ducati | report row 2 : 2 | 0 | 1 | 1.0 | r2 | qatar | losail | 23 february | 23 | xxxx-02-23 | -1 | 2 | 23 | troy corser | fonsi nieto | fonsi nieto | alstare suzuki | report row 3 : 3 | 0 | 2 | 2.0 | r1 | australia | phillip island | 2 march | 2 | xxxx-03-02 | -1 | 3 | 2 | troy bayliss | troy bayliss | troy bayliss | xerox ducati | report row 4 : 4 | 0 | 2 | 2.0 | r2 | australia | phillip island | 2 march | 2 | xxxx-03-02 | -1 | 3 | 2 | troy bayliss | max biaggi | troy bayliss | xerox ducati | report row 5 : 5 | 0 | 3 | 3.0 | r1 | spain | valencia | 6 april | 6 | xxxx-04-06 | -1 | 4 | 6 | max neukirchner | noriyuki haga | lorenzo lanzi | team r.g | report row 6 : 6 | 0 | 3 | 3.0 | r2 | spain | valencia | 6 april | 6 | xxxx-04-06 | -1 | 4 | 6 | max neukirchner | carlos checa | noriyuki haga | yamaha motor italia | report row 7 : 7 | 0 | 4 | 4.0 | r1 | netherlands | assen | 27 april | 27 | xxxx-04-27 | -1 | 4 | 27 | troy bayliss | max neukirchner | troy bayliss | xerox ducati | report row 8 : 8 | 0 | 4 | 4.0 | r2 | netherlands | assen | 27 april | 27 | xxxx-04-27 | -1 | 4 | 27 | troy bayliss | troy bayliss | troy bayliss | xerox ducati | report row 9 : 9 | 0 | 5 | 5.0 | r1 | italy | monza | 11 may | 11 | xxxx-05-11 | -1 | 5 | 11 | troy bayliss | noriyuki haga | max neukirchner | alstare suzuki | report row 10 : 10 | 0 | 5 | 5.0 | r2 | italy | monza | 11 may | 11 | xxxx-05-11 | -1 | 5 | 11 | troy bayliss | noriyuki haga | noriyuki haga | yamaha motor italia | report row 11 : 11 | 0 | 6 | 6.0 | r1 | united states | miller motorsports park | 1 june | 1 | xxxx-06-01 | -1 | 6 | 1 | carlos checa | carlos checa | carlos checa | ten kate honda | report row 12 : 12 | 0 | 6 | 6.0 | r2 | united states | miller motorsports park | 1 june | 1 | xxxx-06-01 | -1 | 6 | 1 | carlos checa | carlos checa | carlos checa | ten kate honda | report row 13 : 13 | 0 | 7 | 7.0 | r1 | germany | nurburgring | 15 june | 15 | xxxx-06-15 | -1 | 6 | 15 | max neukirchner | troy bayliss | noriyuki haga | yamaha motor italia | report row 14 : 14 | 0 | 7 | 7.0 | r2 | germany | nurburgring | 15 june | 15 | xxxx-06-15 | -1 | 6 | 15 |

A: This is a table of racing records of the season, which contains round, country, circuit, date, pole position, \
fastest lap, winning rider, winning team, and report  columns. The row 14 is not complete, thus only \
the part of the table is provided. The last row in the provided incomplete table has nurburgring under the season  column. \
There maybe more records in the original table. So the last race of the season can be something else other than nurburgring. \
Answer B MX Open is not under the season  column in any row. So Answer A is more correct than Answer B.
"""




# nu-714 204_326 A
test_sample_0 ="""
Q: Name the format with the highest value in `` decimal 128 ''.
answer A : exponent range 
answer B : Emax = 3√ó2W-1
 col : id | agg | decimal32 | decimal32_number | decimal64 | decimal64_number | decimal128 | decimal128_number | decimal(32k) | format row 1 : 1 | 0 | 1 | 1.0 | 1 | 1.0 | 1 | 1.0 | 1 | sign field (bits) row 2 : 2 | 0 | 5 | 5.0 | 5 | 5.0 | 5 | 5.0 | 5 | combination field (bits) row 3 : 3 | 0 | 6 | 6.0 | 8 | 8.0 | 12 | 12.0 | w = 2√ók + 4 | exponent continuation field (bits) row 4 : 4 | 0 | 20 | 20.0 | 50 | 50.0 | 110 | 110.0 | t = 30√ók-10 | coefficient continuation field (bits) row 5 : 5 | 0 | 32 | 32.0 | 64 | 64.0 | 128 | 128.0 | 32√ók | total size (bits) row 6 : 6 | 0 | 7 | 7.0 | 16 | 16.0 | 34 | 34.0 | p = 3√ót/10+1 = 9√ók-2 | coefficient size (decimal digits) row 7 : 7 | 0 | 192 | 192.0 | 768 | 768.0 | 12288 | 12288.0 | 3√ó2w = 48√ó4k | exponent range row 8 : 8 | 0 | 96 | 96.0 | 384 | 384.0 | 6144 | 6144.0 | emax = 3√ó2w-1 | largest value is 9.99...√ó10emax row 9 : 9 | 0 | -95 | -95.0 | -383 | -383.0 | -6143 | -6143.0 | emin = 1-emax | smallest normalized value is 1.00...√ó10emin row 10 : 10 | 0 | -101 | -101.0 | -398 | -398.0 | -6176 | -6176.0 | etiny = 2-p-emax | smallest non-zero value is 1√ó10etiny

A:
"""
# GPT4
# ChatCompletionMessage(content='This is a table comparing the qualities of various decimal formats. 
# Based on the table, the highest value in "decimal128"  column is 12288, which corresponds to the "exponent range" 
# in the format row. So, Answer A is the correct one.', role='assistant', function_call=None, tool_calls=None)


# nu-0 203_733 B
test_sample_1="""
Q: which country had the most cyclists finish within the top 10? 
answer A : ita
answer B : Italy
 col : id | agg | rank | rank_number | cyclist | cyclist_first | cyclist_second | team | time | time_number | uci_protour_points | uci_protour_points_number row 1 : 1 | 0 | 1 | 1.0 | alejandro valverde (esp) | alejandro valverde | esp | caisse d'epargne | 5h 29' 10" | 5.0 | 40 | 40.0 row 2 : 2 | 0 | 2 | 2.0 | alexandr kolobnev (rus) | alexandr kolobnev | rus | team csc saxo bank | s.t | nan | 30 | 30.0 row 3 : 3 | 0 | 3 | 3.0 | davide rebellin (ita) | davide rebellin | ita | gerolsteiner | s.t | nan | 25 | 25.0 row 4 : 4 | 0 | 4 | 4.0 | paolo bettini (ita) | paolo bettini | ita | quick step | s.t | nan | 20 | 20.0 row 5 : 5 | 0 | 5 | 5.0 | franco pellizotti (ita) | franco pellizotti | ita | liquigas | s.t | nan | 15 | 15.0 row 6 : 6 | 0 | 6 | 6.0 | denis menchov (rus) | denis menchov | rus | rabobank | s.t | nan | 11 | 11.0 row 7 : 7 | 0 | 7 | 7.0 | samuel sanchez (esp) | samuel sanchez | esp | euskaltel-euskadi | s.t | nan | 7 | 7.0 row 8 : 8 | 0 | 8 | 8.0 | stephane goubert (fra) | stephane goubert | fra | ag2r-la mondiale | + 2" | 2.0 | 5 | 5.0 row 9 : 9 | 0 | 9 | 9.0 | haimar zubeldia (esp) | haimar zubeldia | esp | euskaltel-euskadi | + 2" | 2.0 | 3 | 3.0 row 10 : 10 | 0 | 10 | 10.0 | david moncoutie (fra) | david moncoutie | fra | cofidis | + 2" | 2.0 | 1 | 1.0</s>

A:
"""

# nu-87	203_588 B
test_sample_2="""
Q: what was the first interval of five years to have more than 100,000 deaths? 
answer A : 1965
answer B : 1965-1970
 col : id | agg | period | period_minimum_number | period_minimum_parsed | period_minimum_year | period_maximum_number | period_maximum_parsed | period_maximum_year | live_births_per_year | live_births_per_year_number | deaths_per_year | deaths_per_year_number | natural_change_per_year | natural_change_per_year_number | cbr | cbr_number | cdr | cdr_number | nc | nc_number | tfr | tfr_number | imr | imr_number row 1 : 1 | 0 | 1950-1955 | 1950 | 1950 | 1950 | 1955 | 1955 | 1955 | 139000 | 139000.0 | 66000 | 66000.0 | 74000 | 74000.0 | 52.6 | 52.6 | 24.8 | 24.8 | 27.8 | 27.8 | 6.86 | 6.86 | 174 | 174.0 row 2 : 2 | 0 | 1955-1960 | 1955 | 1955 | 1955 | 1960 | 1960 | 1960 | 164000 | 164000.0 | 76000 | 76000.0 | 88000 | 88000.0 | 53.8 | 53.8 | 24.9 | 24.9 | 29 | 29.0 | 6.96 | 6.96 | 171 | 171.0 row 3 : 3 | 0 | 1960-1965 | 1960 | 1960 | 1960 | 1965 | 1965 | 1965 | 195000 | 195000.0 | 89000 | 89000.0 | 105000 | 105000.0 | 55.5 | 55.5 | 25.5 | 25.5 | 30.1 | 30.1 | 7.13 | 7.13 | 167 | 167.0 row 4 : 4 | 0 | 1965-1970 | 1965 | 1965 | 1965 | 1970 | 1970 | 1970 | 229000 | 229000.0 | 105000 | 105000.0 | 124000 | 124000.0 | 56.2 | 56.2 | 25.8 | 25.8 | 30.4 | 30.4 | 7.32 | 7.32 | 164 | 164.0 row 5 : 5 | 0 | 1970-1975 | 1970 | 1970 | 1970 | 1975 | 1975 | 1975 | 263000 | 263000.0 | 121000 | 121000.0 | 142000 | 142000.0 | 55.8 | 55.8 | 25.6 | 25.6 | 30.2 | 30.2 | 7.52 | 7.52 | 162 | 162.0 row 6 : 6 | 0 | 1975-1980 | 1975 | 1975 | 1975 | 1980 | 1980 | 1980 | 301000 | 301000.0 | 138000 | 138000.0 | 164000 | 164000.0 | 55.1 | 55.1 | 25.1 | 25.1 | 29.9 | 29.9 | 7.63 | 7.63 | 161 | 161.0 row 7 : 7 | 0 | 1980-1985 | 1980 | 1980 | 1980 | 1985 | 1985 | 1985 | 350000 | 350000.0 | 157000 | 157000.0 | 193000 | 193000.0 | 55.4 | 55.4 | 24.8 | 24.8 | 30.6 | 30.6 | 7.76 | 7.76 | 159 | 159.0 row 8 : 8 | 0 | 1985-1990 | 1985 | 1985 | 1985 | 1990 | 1990 | 1990 | 406000 | 406000.0 | 179000 | 179000.0 | 227000 | 227000.0 | 55.9 | 55.9 | 24.6 | 24.6 | 31.3 | 31.3 | 7.81 | 7.81 | 155 | 155.0 row 9 : 9 | 0 | 1990-1995 | 1990 | 1990 | 1990 | 1995 | 1995 | 1995 | 471000 | 471000.0 | 192000 | 192000.0 | 279000 | 279000.0 | 55.5 | 55.5 | 22.7 | 22.7 | 32.8 | 32.8 | 7.78 | 7.78 | 146 | 146.0 row 10 : 10 | 0 | 1995-2000 | 1995 | 1995 | 1995 | 2000 | 2000 | 2000 | 538000 | 538000.0 | 194000 | 194000.0 | 344000 | 344000.0 | 53.5 | 53.5 | 19.3 | 19.3 | 34.2 | 34.2 | 7.6 | 7.6 | 131 | 131.0 row 11 : 11 | 0 | 2000-

A:
"""

# nu-190 203_641 B
test_sample_3="""
Q: what is the last film that lars von trier made? 
answer A : nymphomaniac: volume i
answer B : Nymphomaniac: Volume II
 col : id | agg | year | year_number | film | rotten_tomatoes | rotten_tomatoes_number | metacritic | metacritic_number | imdb | imdb_number | imdb_number1 | imdb_number2 row 1 : 1 | 0 | 1982 | 1982 | images of liberation | None | nan | None | nan | 5.1/10 | 5.1 | 5.1 | 10 row 2 : 2 | 0 | 1984 | 1984 | the element of crime | 77% | 77.0 | None | nan | 6.9/10 | 6.9 | 6.9 | 10 row 3 : 3 | 0 | 1987 | 1987 | epidemic | 33% | 33.0 | 66% | 66.0 | 6.1/10 | 6.1 | 6.1 | 10 row 4 : 4 | 0 | 1991 | 1991 | europa | 85% | 85.0 | 66% | 66.0 | 7.7/10 | 7.7 | 7.7 | 10 row 5 : 5 | 0 | 1996 | 1996 | breaking the waves | 86% | 86.0 | 76% | 76.0 | 7.9/10 | 7.9 | 7.9 | 10 row 6 : 6 | 0 | 1998 | 1998 | the idiots | 70% | 70.0 | 47% | 47.0 | 6.9/10 | 6.9 | 6.9 | 10 row 7 : 7 | 0 | 2000 | 2000 | dancer in the dark | 68% | 68.0 | 61% | 61.0 | 8.0/10 | 8.0 | 8.0 | 10 row 8 : 8 | 0 | 2003 | 2003 | the five obstructions | 88% | 88.0 | 79% | 79.0 | 7.5/10 | 7.5 | 7.5 | 10 row 9 : 9 | 0 | 2003 | 2003 | dogville | 70% | 70.0 | 59% | 59.0 | 8.0/10 | 8.0 | 8.0 | 10 row 10 : 10 | 0 | 2005 | 2005 | manderlay | 51% | 51.0 | 46% | 46.0 | 7.4/10 | 7.4 | 7.4 | 10 row 11 : 11 | 0 | 2006 | 2006 | the boss of it all | 74% | 74.0 | 71% | 71.0 | 6.7/10 | 6.7 | 6.7 | 10 row 12 : 12 | 0 | 2009 | 2009 | antichrist | 48% | 48.0 | 49% | 49.0 | 6.6/10 | 6.6 | 6.6 | 10 row 13 : 13 | 0 | 2011 | 2011 | melancholia | 77% | 77.0 | 80% | 80.0 | 7.1/10 | 7.1 | 7.1 | 10 row 14 : 14 | 0 | 2013 | 2013 | nymphomaniac: volume i | 77% | 77.0 | 63% | 63.0 | 7.5/10 | 7.5 | 7.5 | 10 row 15 : 15 | 0 | 2013 | 2013 | nymphomaniac: volume ii | 79% | 79.0 | 76% | 76.0 | 7.2/10 | 7.2 | 7.2 | 10

A:
"""


# nu-27 203_307 A
test_sample_4="""
Q: who was the top scorer in the last season? 
answer A : simon makienok christoffersen (10)
answer B : Bent Christensen
 col : id | agg | season | season_minimum_number | season_minimum_parsed | season_minimum_year | season_maximum_number | season_maximum_parsed | season_maximum_year | league_pos. | league_pos._number | league_competition | league_top_scorer | league_top_scorer_first | league_top_scorer_second | league_top_scorer_second_number | danish_cup | europe | europe_list | europe_length | others | others_list | others_length row 1 : 1 | 0 | 1981-82 | 1981 | 1981 | 1981 | 1982 | 1982 | 1982 | 4 | 4.0 | 1982 1st division | michael laudrup (15) | michael laudrup | 15 | 15.0 | 4th round | None | | 0 | None | | 0 row 2 : 2 | 0 | 1982-83 | 1982 | 1982 | 1982 | 1983 | 1983 | 1983 | 4 | 4.0 | 1983 1st division | brian chr√∏is (12) | brian chr√∏is | 12 | 12.0 | 4th round | None | | 0 | None | | 0 row 3 : 3 | 0 | 1983-84 | 1983 | 1983 | 1983 | 1984 | 1984 | 1984 | 4 | 4.0 | 1984 1st division | jens kolding (11) | jens kolding | 11 | 11.0 | 3rd round | None | | 0 | None | | 0 row 4 : 4 | 0 | 1984-85 | 1984 | 1984 | 1984 | 1985 | 1985 | 1985 | 1 | 1.0 | 1985 1st division | claus nielsen (17) | claus nielsen | 17 | 17.0 | 3rd round | None | | 0 | None | | 0 row 5 : 5 | 0 | 1985-86 | 1985 | 1985 | 1985 | 1986 | 1986 | 1986 | 2 | 2.0 | 1986 1st division | claus nielsen (16) | claus nielsen | 16 | 16.0 | quarter-final | None | | 0 | None | | 0 row 6 : 6 | 0 | 1986-87 | 1986 | 1986 | 1986 | 1987 | 1987 | 1987 | 1 | 1.0 | 1987 1st division | claus nielsen (20) | claus nielsen | 20 | 20.0 | 4th round | ec1 quarter-final | ec1 quarter-final | 1 | None | | 0 row 7 : 7 | 0 | 1987-88 | 1987 | 1987 | 1987 | 1988 | 1988 | 1988 | 1 | 1.0 | 1988 1st division | bent christensen (21) | bent christensen | 21 | 21.0 | finalist | ec3 2nd round | ec3 2nd round | 1 | None | | 0 row 8 : 8 | 0 | 1988-89 | 1988 | 1988 | 1988 | 1989 | 1989 | 1989 | 2 | 2.0 | 1989 1st division | bent christensen (10) | bent christensen | 10 | 10.0 | winner | ec1 1st round | ec1 1st round | 1 | None | | 0 row 9 : 9 | 0 | 1989-90 | 1989 | 1989 | 1989 | 1990 | 1990 | 1990 | 1 | 1.0 | 1990 1st division | bent christensen (17) | bent christensen | 17 | 17.0 | quarter-final | ec1 1st round | ec1 1st round | 1 | None | | 0 row 10 : 10 | 0 | 1990-91 | 1990 | 1990 | 1990 | 1991 | 1991 | 1991 | 1 | 1.0 | 1991 superliga | bent christensen (11) | bent christensen | 11 | 11.0 | semi-final | ec3 semi-final | ec3 semi-final | 1 | None | | 0 row 11 : 11 | 0 | 1991-92 | 1991 | 1991 | 1991 | 1992 | 1992 | 1992 | 7 | 7.0 | 1991-92 superliga | kim vilfort (9) | kim vilfort | 9 | 9.0 | 4th round | ec1 2nd round | ec1 2nd round | 1 | None | | 0 row 12 : 12 | 0 | 1992-93 | 1992 | 1992 | 1992 | 1993 | 1993 | 1993 | 3 | 3.0 | 1992-93 superliga | kim vilfort (10) | kim vilfort | 10 | 10.0 | 5th round | None | | 0 | None | | 0 row 13 :

A:
"""

# nu-14	203_128 B
test_sample_5="""
Q: what is the only character with a blank c string? 
answer A : l|l
answer B : Space
 col : id | agg | name | glyph | c_string | unicode | unicode_name | unicode_name_first | unicode_name_second row 1 : 1 | 0 | nul | None | \0 | u+0000 | null (nul) | null | nul row 2 : 2 | 0 | alert | None | \a | u+0007 | bell (bel) | bell | bel row 3 : 3 | 0 | backspace | None | \b | u+0008 | backspace (bs) | backspace | bs row 4 : 4 | 0 | tab | None | \t | u+0009 | character tabulation (ht) | character tabulation | ht row 5 : 5 | 0 | carriage-return | None | \r | u+000d | carriage return (cr) | carriage return | cr row 6 : 6 | 0 | newline | None | \n | u+000a | line feed (lf) | line feed | lf row 7 : 7 | 0 | vertical-tab | None | \v | u+000b | line tabulation (vt) | line tabulation | vt row 8 : 8 | 0 | form-feed | None | \f | u+000c | form feed (ff) | form feed | ff row 9 : 9 | 0 | space | None | None | u+0020 | space | space | None row 10 : 10 | 0 | exclamation-mark |! |! | u+0021 | exclamation mark | exclamation mark | None row 11 : 11 | 0 | quotation-mark | " | \" | u+0022 | quotation mark | quotation mark | None row 12 : 12 | 0 | number-sign | # | # | u+0023 | number sign | number sign | None row 13 : 13 | 0 | dollar-sign | $ | $ | u+0024 | dollar sign | dollar sign | None row 14 : 14 | 0 | percent-sign | % | % | u+0025 | percent sign | percent sign | None row 15 : 15 | 0 | ampersand | & | & | u+0026 | ampersand | ampersand | None row 16 : 16 | 0 | apostrophe |'| \' | u+0027 | apostrophe | apostrophe | None row 17 : 17 | 0 | left-parenthesis | ( | ( | u+0028 | left parenthesis | left parenthesis | None row 18 : 18 | 0 | right-parenthesis | ) | ) | u+0029 | right parenthesis | right parenthesis | None row 19 : 19 | 0 | asterisk | * | * | u+002a | asterisk | asterisk | None row 20 : 20 | 0 | plus-sign | + | + | u+002b | plus sign | plus sign | None row 21 : 21 | 0 | comma |, |, | u+002c | comma | comma | None row 22 : 22 | 0 | hyphen | - | - | u+002d | hyphen-minus | hyphen-minus | None row 23 : 23 | 0 | period |. |. | u+002e | full stop | full stop | None row 24 : 24 | 0 | slash | / | / | u+002f | solidus | solidus | None row 25 : 25 | 0 | zero | 0 | 0 | u+0030 | digit zero | digit zero | None row 26 : 26 | 0 | one | 1 | 1 | u+0031 | digit one | digit one | None row 27 : 27 | 0 | two | 2 | 2 | u+0032 | digit two | digit two | None row 28 : 28 | 0 | three | 3 | 3 | u+0033 | digit three | digit three | None row 29 : 29 | 0 | four | 4 | 4 | u+0034 | digit four | digit four | None row 30 : 30 | 0 | five | 5 | 5 | u+0035 | digit five | digit five | None row 31 : 31 | 0 | six | 6 | 6 | u+0036 | digit six | digit six | None row 32 : 32 | 0 | seven | 7 | 7 | u+0037 | digit seven | digit seven | None row 33 : 33 | 0 | eight | 8 | 8 | u+0038 | digit eight | digit eight | None row 34 : 34 | 0 | nine | 9 | 9 | u+0039 | digit nine | digit nine | None row 35 : 35 | 0 |  colon | : | : | u+003a |  colon |  colon | None row 36 : 36 | 0 | semi colon |

A:
"""


import json
import sqlite3
import requests

def tmp(data):
    # whether it's communter train or not
    r = requests.get('https://api-v3.mbta.com/routes/{}'.format(data['route_id']), params={'api_key': '1507e676a22b462f9a7c380b1f1222dc'})
    r = r.json()
    eps = 0.000001
    if not r['data']['attributes']['fare_class'] == 'Commuter Rail' or data['speed'] - 10 < eps:
        return
    conn = sqlite3.connect('./sqlite.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM train_lookup')
    train_rows = cursor.fetchall()
    train_found = None
    for row in train_rows:
        if data['vehicle_id'] == row[1]:
            train_found = True
            continue
    if not train_found:
        cursor.execute("INSERT INTO train_lookup (train_id, line_name) VALUES ('{}', '{}')".format(
            data['vehicle_id'],
            r['data']['attributes']['long_name']
        ))
        conn.commit()
    else:
        cursor.execute("UPDATE train_lookup SET sightings = {} WHERE train_id = '{}'".format(
            row[3] + 1,
            data['vehicle_id'],
        ))
        conn.commit()
    cursor.execute("INSERT INTO train_log (train_id, latitude, longitude, speed) VALUES ('{}', '{}', '{}', '{}');".format(
        data['vehicle_id'],
        data['latitude'],
        data['longitude'],
        data['speed'],
    ))
    conn.commit()



# 1. Save a log of speeding incidents to the database. Only log incidents for trains that meet both of the following criteria:
#     - The trai‍‍‌‌‍‌‌‌‌‌‍‍‌‌‌‍‌‍‍n must be a commuter rail train.
#     - The train must be going over 10 miles per hour.
# 2. Know how many times a specific train has been speeding and where.
#     - You will not need to implement this function, but your data model must be able to satisfy this requirement.
# 面试题是给段代码， 实现功能，提出scalability 和 inefficient 方面的建议

# 先是HR聊, coding round是个collaborative coding，和两个engineer一起，然后debug一个已经写好的code，需要解释现在的code有什么问题，
# 需要怎么optimize. 题目在地里已经有了就不发了，就是那个train的问题。重点在于database transaction 需要怎么写，SQL有什么问题。
# 挺简单的，需要注意communication, 然后理解example的API。
# 然后是和hiring manager聊，也是45分钟。挺常规的介绍一下自己做过的东西，然后主要是问了两个问题，一个是如果让你deliver一个feature, 
# 你是会慢慢design还是赶快deliver. 另一个问题是你是怎么ma‍‍‌‌‍‌‌‌‌‌‍‍‌‌‌‍‌‍‍nage做过的一个project的priority. 聊的挺好的，第二天收到schedule onsite.


# 面试的体验非常好，第一轮是一个collaborative working simulation，类似于你对着一个project来debug然后加一些功能，
# 有两个人看着你，有不懂的可以直接问
# final就是algo+collaborate+bq，algo难度是一个easy和medi‍‍‌‌‍‌‌‌‌‌‍‍‌‌‌‍‌‍‍um中间的，不是那种5分钟就写完的easy，但也没难到medium的平均水平
#（对地里大多数人来说应该都是能10分钟秒的水平)


# 一开始先介绍一下project是什么，过一下summary文档。然后让我自己看代码按照summary里面的要求去改动，不懂的问题可以问，
# 他们觉得能回答的就会回答你。我主要问的是API的问题，有些sql有关的method并不熟悉。最后并不需要跑起来。
# 给的project是一个web application。主要workflow是收到request，生成数据库query, 然后根据query得到的结果进行运算，
# 把得到的结果拿去更新数据库。改了几个明显的问题以后，就开‍‍‌‌‍‌‌‌‌‌‍‍‌‌‌‍‌‍‍始followup，问些scalability，reliability的问题。
# 代码可以改进的点都挺明显的。同时考察你自己熟悉的编程语言，SQL和一点网络知识。


# In [18]: flatten({'a': {'b':'foo', 'c': [4,5,6]}} )
# Out[18]: [('a.b', 'foo'), ('a.c', 4), ('a.c', 5), ('a.c', 6)]


# 一共三轮，第一轮 做题，主题就是refactor python code. 第二轮hr跟你聊. 最后一轮就是视频面试，还是做题，主题还是refactor python code
# 最后一轮就是给了份python file, 问你怎么refactor最后，‍‍‌‌‍‌‌‌‌‌‍‍‌‌‌‍‌‍‍我差不多给出了6，7处可以提高的地方，但中途确实表现的紧张了，导致人家fail了我

# Be familiar with http servers and clients , as well as basic SQL commands.
# JSON manipulation. , design problems , recursion

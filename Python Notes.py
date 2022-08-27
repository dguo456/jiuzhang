"""
Avoid: W.E.T (Wrote Everything Twice) Comments
  Use: D.R.Y (Don't Repeat Yourself) Comments

By using obvious naming conventions, we are able to remove all unnecessary comments and reduce the length of the code as well!
Your comments should rarely be longer than the code they support.
String and Tuple are immutable, Tuples are faster than lists
The main advantage of tuples is that tuples can be used as keys in dictionaries, while lists can't

nums_squared_lc = [num**2 for num in range(5)]    ---     list comprehension
nums_squared_gc = (num**2 for num in range(5))    ---     return a generator
"""
# check if a number is a palindrome
def is_palindrome(num):
    # Skip single-digit inputs
    if num // 10 == 0:
        return False
    temp = num
    reversed_num = 0

    while temp != 0:
        reversed_num = (reversed_num * 10) + (temp % 10)
        temp = temp // 10

    if num == reversed_num:
        return num
    else:
        return False


# Python map() function
# Example 1
def addition(n):
    	return n + n

numbers = (1, 2, 3, 4)
result = map(addition, numbers)
print(list(result))

# Example 2
numbers = (1, 2, 3, 4)
result = map(lambda x: x + x, numbers)
print(list(result))

# Example 3
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
result = map(lambda x, y: x + y, numbers1, numbers2)
print(list(result))

# Example 4
l = ['sat', 'bat', 'cat', 'mat']
test = list(map(list, l))       # [['s', 'a', 't'], ['b', 'a', 't'], ['c', 'a', 't'], ['m', 'a', 't']]


# map(), filter(), reduce() with lambda function
"""
All three of these can be replaced with List Comprehension
lambda is an anonymous function/method, and it will be used only once 
Lambdas differ from normal Python methods because they can have only one expression, can't contain any statements and their return type is a function object.
Lambda 与普通 Python 方法不同，因为它们只能有一个表达式，不能包含任何语句，并且它们的返回类型是函数对象
Since all three of these methods expect a function object as the first argument, a lambda function here is useful
"""
# Examples of lambda function, map() function, filter() function and reduce() function
fruit = ["Apple", "Banana", "Pear", "Apricot", "Orange"]
map_list = list(map(lambda s: s[0] == "A", fruit))              # [True, False, False, True, False]
filter_object = list(filter(lambda s: s[0] == "A", fruit))      # ['Apple', 'Apricot']
from functools import reduce
from socket import socket
list = [2, 4, 7, 3]
print(reduce(lambda x, y: x + y, list))                                             # 16
print("With an initial value: " + str(reduce(lambda x, y: x + y, list, 10)))        # With an initial value: 26


# Reverse a list
aList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Example 1. reverse list in place
aList.reverse()
# Example 2. reverse list and create a new reversed_list
reversed_list = reversed(aList)
# Example 3. reverse list through slicing
reversed_list = aList[len(aList)-1:-1:-1]
reversed_list = aList[::-1]
# Example 4. list comprehension
reversed_aList = [aList[i] for i in range(len(aList)-1,-1,-1)]
# Example 5. sort list in reversed order
aList.sort(reverse=True)


# Some List Methods
# Example 1. Pop and Append, both are applied to the last element (by default), list.pop() equals list.pop(-1)
# Example 2. Extend
lst = [42,98,77]
lst2 = [8,69]
lst.append(lst2)  # [42, 98, 77, [8, 69]]
lst.extend(lst2)  # [42, 98, 77, 8, 69]
# Example 3. Remove
# Example 4. Find the position of an element using index()
colours = ["red", "green", "blue", "green", "yellow"]
colours.index("green")          # 1
colours.index("green", 2)       # 3, 2 is the starting index
colours.index("green", 3, 4)    # 3, (3, 4) is the starting and ending index
# Example 5. Insert
lst = ["German is spoken", "in Germany,", "Austria", "Switzerland"]
lst.insert(3, "and")  # ['German is spoken', 'in Germany,', 'Austria', 'and', 'Switzerland']


# Shallow Copy and Deep Copy
colours1 = ["red", "blue"]
colours2 = colours1  # Shallow, id(colours1) == id(colours2), both are references to the same object
colours2[1] = "green"  # print(colours1) returns ['red', 'green']
from copy import deepcopy
colours2 = deepcopy(colours1)  # Deepcopy, 


# Dictionary Methods
# Example 1. popitem() and get()
proj_language = {"proj1":"Python", "proj2":"Perl", "proj3":"Java", "proj4":"Python"}
proj_language.popitem()  # popout the last (key, value) pair
proj_language.get("proj4", "Python")  # using get() to access Non-existing Keys and set a default value
d1, d2 = dict(a=4, b=5, d=8), dict(a=1, d=10, e=9)
merge_sum = { k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2) }


# Set Methods
# Example 1. add(), difference(), discard(): nothing happen if element not in set, remove(): return error if element not in set
adjectives = {"cheap","expensive","inexpensive","economical"}
adjectives  # return an ordered set {'cheap', 'economical', 'expensive', 'inexpensive'}
adjectives.add("Python")
adjectives.difference({"a", "b", "c"})  # return {'expensive', 'economical', 'inexpensive', 'cheap'}
adjectives - {"a", "b", "c"}  # same as above
adjectives.difference_update({"a", "b", "c"})  # return {'expensive', 'economical', 'inexpensive', 'cheap'}

# Example 2. union(), intersection()
adjectives.union({"Python", "Java"})  # return {'Python', 'expensive', 'economical', 'inexpensive', 'cheap', 'Java'}
adjectives | {"Python", "Java"}  # same above
adjectives.intersection({"Python", "cheap", "expensive"})  # {'expensive', 'cheap'}

# Example 3. isdisjoint(), issubset(), issuperset(), pop()
x = {"a","b","c"}
y = {"c","d","e"}
x.isdisjoint(y)  # False, returns True if two sets have a null intersection.
x.issubset(y)  # False, returns True if x is a subset of y, equals to x < y
x.pop()  # return c



# Optional Arguments in Python With *args and **kwargs
class Car:
    def __init__(self, color, mileage):
        self.color = color
        self.mileage = mileage

class AlwaysBlueCar(Car):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = 'blue'

car = Car('red', 8000)
print(car.color)                        # red
blue_car = AlwaysBlueCar('red', 8000)
blue_car.color                          # 'blue'


# Optional Arguments and Function Argument Unpacking
def foo(required, *args, **kwargs):
    print(required)
    if args:
        print(args)
    if kwargs:
        print(kwargs)

foo('hello', 1, 2, 3, key1='value1', key2=999)  # This will print: hello   /n  (1, 2, 3)   /n   {'key1': 'value1', 'key2': 999}

tuple_vec = (1, 2, 3)
dict_vec = {'x': 1, 'y': 2, 'z': 3}
def print_vector(x, y, z):
    print('<%s, %s, %s>' % (x, y, z))

print_vector(*tuple_vec)    # <1, 2, 3>
print_vector(*dict_vec)     # <x, y, z>
print_vector(**dict_vec)    # <1, 2, 3>




# formatted string literals (f-Strings)
class Comedian:
    def __init__(self, first_name, last_name, age):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age

    def __str__(self):
        return f"{self.first_name} {self.last_name} is {self.age}."

    # Always at least create a __repr__ function in your codebase
    def __repr__(self):
        return f"{self.first_name} {self.last_name} is {self.age}. Surprise!"
    
new_comedian = Comedian("Eric", "Idle", "74")
f"{new_comedian}"  # 'Eric Idle is 74.'
f"{new_comedian!r}"  # 'Eric Idle is 74. Surprise!', with the conversion flag !r -> call __repr__    flag !s -> call __str__
f"The \"comedian\" is {'Eric Idle'}, aged {'74'}."  # 'The "comedian" is Eric Idle, aged 74.'

# Multiline f-string, the \ can be removed, will have the same format
name, profession, affiliation = "Eric", "comedian", "Monty Python"
message = (
            f"Hi {name}. " \
            f"You are a {profession}. " \
            f"You were in {affiliation}."
)
message


# Make custom classes orderable
"""min(), max(), and sorted all need the objects to be orderable. The class needs to define all of the 6 methods __lt__, __gt__, __ge__, __le__, __ne__ and __eq__"""
class IntegerContainer(object):
    def __init__(self, value):
        self.value = value
        
    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.value)
    
    def __lt__(self, other):
        print('{!r} - Test less than {!r}'.format(self, other))
        return self.value < other.value
    
    def __le__(self, other):
        print('{!r} - Test less than or equal to {!r}'.format(self, other))
        return self.value <= other.value

    def __gt__(self, other):
        print('{!r} - Test greater than {!r}'.format(self, other))
        return self.value > other.value

    def __ge__(self, other):
        print('{!r} - Test greater than or equal to {!r}'.format(self, other))
        return self.value >= other.value

    def __eq__(self, other):
        print('{!r} - Test equal to {!r}'.format(self, other))
        return self.value == other.value

    def __ne__(self, other):
        print('{!r} - Test not equal to {!r}'.format(self, other))
        return self.value != other.value


# 好习惯系列

# 1. don't manually calling close on a file
def test(filename):
    with open(filename) as f:
        f.write("hello!\n")

    with open('data.json') as f:
        data = json.load(f)

# 2. use context manager instead of using 'finally'
def test(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(b'Hello, world')

# 3. Don't use bare 'except'
def test():
    while True:
        try:
            s = input("Input a number: ")
            x = int(s)
            break
        except ValueError:
            print("Not a number, please try again")

# 4. Different Comprehensions
list_comp = [i*i for i in range(10)]
set_comp = {i%3 for i in range(10)}
dict_comp = {i: i*i for i in range(10)}
gen_comp = (2*x+5 for x in range(10))

def test(a, b, n):
    c = []
    for i in range(n):
        for j in range(n):
            ij_entry = sum(a[n * i + k] * b[n * k + j] for k in range(n))
            c.append(ij_entry)

# 5. check variable's type, don't use a == tuple, use 'isinstance' method
def test():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    if isinstance(p, tuple):
        print("it is a tuple")

# 6. use 'enumerate' and 'zip'
def test():
    a, b = [1, 2, 3], [4, 5, 6]
    for i, (av, bv) in enumerate(zip(a, b)):
        ...

# 7. use logging modele instead of print the message
import logging
def main():
    level = logging.DEBUG
    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=level, format=fmt)

# 8. Python is a interpreted code but also a compiled code, instead of compiling to the machine code, 
# it's actually compiled to the bytecode. That bytecode is then run by the interpreter



# Tuple Unpacking
# A tuple isn’t just an immutable list. So it can also be used as record-keeping purpose.
fruits = [('banana', 'yellow', 10), ('apple', 'red', 19)]
for name, _, _ in fruits:
    print(name)

# Function returning the sum of its arguments
def add(a, b):
  return a+b

numbers = (1, 2)
result = add (*numbers) # Unpacking numbers with a * operator
print(result)




# Namedtuple
"""
A namedtuple is another container sequence but an extended version of the built-in tuple sequence. It is also an immutable sequence.
Limitations of a tuple:
1. We can only access data from a tuple by using an index (that is not a human-readable identifier).
2. Secondly, there is no guarantee that tuples that are supposed to hold the same type of values will do so. 
You can easily make a list of tuples containing different types of values. It makes debugging difficult.

Tuple vs Namedtuple: about the same amount of time spent using indices but Namedtuple has more readability! Dictionary lies between a plain tuple and a namedtupled 
in terms of performance and readability. Named tuples clearly win in readability but lag in creation and access times. Plain tuples are fast to create and access 
but using indices 0, 1, 2 makes my head spin and I can actually mix them up.
"""
from collections import namedtuple  #Importing namedtuple

Fruit = namedtuple('Fruit', 'name color price')   # Made namedtuple of type Fruit

# Creating objects of type Fruit
f1 = Fruit('apple', 'red', 29)
f2 = Fruit('banana', 'yellow', 19)
print(f1.name, f1.color, f1.price)      # Same with print(f1[0], f1[1], f1[2])
print(f2.name, f2.color, f2.price)      # Same with print(f2[0], f2[1], f2[2])

x = namedtuple('x', 'a, b, c')
print(x._fields)                        # The _fields attribute, no need to scheck back what the fields are. will return ('a', 'b', 'c')

iterable = ['i', 'j', 'k']
y = x._make(iterable)                   # The _make(iterable) class method, create an instance of the iterable of type x.  y = ('i', 'j', 'k'), return x(a='i', b='j', c='k')

y = x('i', 'j', 'k')
z = y._asdict()                         # The _asdict() instance method, will create dictionary, return OrderedDict([('a', 'i'), ('b', 'j'), ('c', 'k')])
y = y._replace(b='l')                   # The _replace(**kwargs) instance method: Replacing the old value, return x(a='i', b='l', c='k') 



# Stack vs Queue
"""
A stack is a data structure that follows the LIFO (Last In First Out) order for push (insertion) and pop (deletion) functions.
A queue is a data structure that follows the FIFO (First In First Out) order for enqueue (insertion) and dequeue (deletion) functions.
A list does make a good stack, because append() adds an element at the end and pop() deletes an element from the end, in amortized time of O(1).
But list is a terrible choice to make a queue because inserting at and deleting from the beginning requires shifting all the elements by one, requiring O(n) time.
Using collections.deque: deque is like a double-ended queue that lets you add or remove elements from either end. It takes O(1) time for both operations.
But backend implementation of deque is a doubly-linked list, which is why random access in the worst case is O(n) unlike the list.
"""

# Prefix to Postfix Conversion
from collections import deque

def postfix(prefix):
    stack = deque()
    prefix = ''.join((reversed(prefix)))    #Reversing the expression
    for symbol in prefix:
        if symbol != '+' and symbol != '-' and symbol != '*' and symbol != '/':
            stack.append(symbol)    # Push if an operand
        else:
            # If an operator, then pop two operands from stack
            operand1 = stack.pop()
            operand2 = stack.pop()
            # Concatenate operands and operator and again push it in stack
            stack.append(operand1+operand2+symbol)
    return(stack.pop()) # Poping the result

print(postfix('*-A/BC+/AKL'))       # ABC/-AK/L+*



"""
A decorator is a function that takes another function as an argument, does some actions, and then returns the argument based on the actions performed. 
Since functions are first-class object in Python, they can be passed as arguments to another functions.
Hence a decorator is a callable that accepts and returns a callable. Any object which implements the special method __call()__ is termed as callable
"""
# Example 1         Multi Decrators
def reverse_decorator(function):
    def reverse_wrapper():
        make_reverse = "".join(reversed(function()))
        return make_reverse

    return reverse_wrapper
  
def uppercase_decorator(function):
    def uppercase_wrapper():
        var_uppercase = function().upper()
        return var_uppercase
  
    return uppercase_wrapper

# the order of the decorator applied is important, which is applied from bottom to top
@uppercase_decorator
@reverse_decorator      
def say_hi():
    return 'hi george'
  
def main():
    print(say_hi())
  
if __name__ == "__main__":
    main()      # EGROEG IH


# Example 2         Error Handling
def Error_Handler(func):
    def Inner_Function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypeError:
            print(f"{func.__name__} wrong data types. enter numeric")
    return Inner_Function

@Error_Handler
def Mean(a,b):
        print((a+b)/2)
     
@Error_Handler
def Square(sq):
        print(sq*sq)
 
@Error_Handler
def Divide(l,b):
        print(b/l)
     
Mean(4,5)               # 4.5 
Square("three")         # Square wrong data types. enter numeric 
Divide("two","one")     # Divide wrong data types. enter numeric
Mean("six","five")      # Mean wrong data types. enter numeric



# Example 3     Memorization
memory = {}
def memoize_factorial(f):
    def inner(num):
        if num not in memory:
            memory[num] = f(num)
            print('result saved in memory')
        else:
            print('returning result from saved memory')
        return memory[num]
 
    return inner
     
@memoize_factorial
def facto(num):
    if num == 1:
        return 1
    else:
        return num * facto(num-1)
 
print(facto(5))
print(facto(5)) # directly coming from saved memory



# Data Class & namedtuple
from dataclasses import dataclass, field, astuple
from typing import List

# Data-Class  VS  Regular-Class
@dataclass
class DataClassCard:
    rank: str
    suit: str

class RegularCard:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(rank={self.rank!r}, suit={self.suit!r})')  # !r is a regex

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.rank, self.suit) == (other.rank, other.suit)


queen_of_hearts = DataClassCard('Q', 'Hearts')
queen_of_hearts                                  # DataClassCard(rank='Q', suit='Hearts')
queen_of_hearts.rank                             # 'Q'
queen_of_hearts == DataClassCard('Q', 'Hearts')  # True

queen_of_hearts_regular = RegularCard('Q', 'Hearts')
queen_of_hearts_regular                                  # RegularCard(rank='Q', suit='Hearts')
queen_of_hearts_regular.rank                             # 'Q'
queen_of_hearts_regular == DataClassCard('Q', 'Hearts')  # False


# Method.2     Data Class: order=True will enable the comparison, frozon=True, __post_init__()
@dataclass(order=True)
class Person:
    first_name: str = "Dan"
    last_name: str = "Guo"
    age: int = 33
    job: str = "Software Engineer"
    full_name: str = field(init=False, repr=True)
    sort_index: int = field(init=False, repr=False)

    # some field depend on other fields' value, like full_name need first and last name, so set field(init=False)
    def __post_init__(self,):
        self.full_name = self.first_name + " " + self.last_name
        self.sort_index = self.age

    # Since DataClass are not iterable (not like namedtuple), we need to define and return an iterator
    def __iter__(self,):
        return iter(astuple(self))

dan = Person()
dan                  # Person(first_name='Dan', last_name='Guo', age=33, job='Software Engineer', full_name='Dan Guo')
dan.full_name        # Dan Guo

p1 = Person(age=30)
p2 = Person(age=20)
print(p1 > p2)              # True

for field in Person("Dan", "Guo", 33, "SDE"):
    print(field)


# Method.3      Subclassing namedtuple Classes
"""
Data Classes can be thought of as “mutable namedtuples with defaults.”
data classes are like mutable named tuples with type hints.
"""
from collections import namedtuple
from datetime import date

BasePerson = namedtuple(
    "BasePerson",
    "name birthday country",
    defaults=["China"]
)

class Person(BasePerson):
    """A namedtuple subclass to hold a person's data."""

    # set __slots__ to an empty tuple prevents the automatic creation of a per-instance .__dict__. This keeps your BasePerson subclass memory efficient.
    __slots__ = ()

    # add a custom .__repr__() to provide a nice string representation for the class
    def __repr__(self):
        return f"Name: {self.name}, age: {self.age} years old."
    
    # add a property to compute the person's age using datetime
    @property
    def age(self):
        return (date.today() - self.birthday).days // 365

Person.__doc__      # "A namedtuple subclass to hold a person's data."
Dan = Person("Dan", date(1988, 5, 10))
print(Dan)          # Name: Dan, age: 33 years old.
print(Dan.age)      # 33



"""
Factory Method

The "single responsibility principle" states that a module, a class, or even a method should have a single, well-defined responsibility. 
It should do just one thing and have only one reason to change.

Code Refactoring: Improving the Design of Existing Code 
      Defination: “the process of changing a software system in such a way that does not alter the external behavior of the code yet improves its internal structure.”

Factory Method should be used in every situation where an application (client) depends on an interface (product) 
    to perform a task and there are multiple concrete implementations of that interface. 
    You need to provide a parameter that can identify the concrete implementation and use it in the creator to decide the concrete implementation.
"""
# In serializer_demo.py

import json
import xml.etree.ElementTree as et

# The mechanics of Factory Method are always the same. A client (SongSerializer.serialize()) depends on a concrete implementation of an interface. 
# It requests the implementation from a creator component (get_serializer()) using some sort of identifier (format).
# The creator returns the concrete implementation according to the value of the parameter to the client, and the client uses the provided object to complete its task.
class Song:
    def __init__(self, song_id, title, artist):
        self.song_id = song_id
        self.title = title
        self.artist = artist

class SongSerializer:
    # This .serialize() method is the application code that depends on an interface to complete its task.
    # Client Component - a function that takes a Song and returns a string representation.
    def serialize(self, song, format):
        serializer = self._get_serializer(format)
        return serializer(song)

    # This method does not call the concrete implementation, it only returns the function object itself.
    # Creator component - decides which concrete implementation to use.
    def _get_serializer(self, format):
        if format == 'JSON':
            return self._serialize_to_json
        elif format == 'XML':
            return self._serialize_to_xml
        else:
            raise ValueError(format)

    # Concrete implementation 1
    def _serialize_to_json(self, song):
        payload = {
            'id': song.song_id,
            'title': song.title,
            'artist': song.artist
        }
        return json.dumps(payload)

    # Concrete implementation 2
    def _serialize_to_xml(self, song):
        song_element = et.Element('song', attrib={'id': song.song_id})
        title = et.SubElement(song_element, 'title')
        title.text = song.title
        artist = et.SubElement(song_element, 'artist')
        artist.text = song.artist
        return et.tostring(song_element, encoding='unicode')


# from the main.py
import serializer_demo as sd
song = sd.Song('1', 'Water of Love', 'Dire Straits')
# Product component - interface
serializer = sd.SongSerializer()

serializer.serialize(song, 'JSON')  # '{"id": "1", "title": "Water of Love", "artist": "Dire Straits"}'
serializer.serialize(song, 'XML')   # '<song id="1"><title>Water of Love</title><artist>Dire Straits</artist></song>'
serializer.serialize(song, 'YAML')  # ValueError: YAML



# python, heapq: difference between heappushpop() and heapreplace()
from heapq import *
a = [2,7,4,0,8,12,14,13,10,3,4]
heapify(a)
b = a[:]
heappushpop(a, -1)          #   -1  heapreplace(a, x) pushes x onto a first and then pop the smallest value
heapreplace(b, -1)          #   0   heappushpop(a, x) pop the smallest value first and then pushes x onto a





# How to define a Point Class
"""
Creating and initializing objects of a given class is a fundamental step in object-oriented programming. 
This step is often referred to as object construction or instantiation. The tool responsible for running this instantiation process is commonly known as a class constructor.
Class constructors internally trigger Python's instantiation process, which runs through two main steps: instance creation and instance initialization.
Python's instantiation process starts with a call to the class constructor, which triggers the instance creator, .__new__(), to create a new empty object. 
The process continues with the instance initializer, .__init__(), which takes the constructor's arguments to initialize the newly created object.
"""
class Point:
    def __new__(cls, *args, **kwargs):      # Double Underscore (Dunder) -- Dunder methods let you add extra properties to your classes
        print("1. Create a new instance of Point.")
        return super().__new__(cls)

    def __init__(self, x, y):       # Contructor
        print("2. Initialize the new instance of Point.")
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"{type(self).__name__}(x={self.x}, y={self.y})"

    for _ in range(10): pass         # 为了好看


"""
Using global variables in a function
You can use a global variable within other functions by declaring it as global within each function that assigns a value to it:
"""
globvar = 0

def set_globvar_to_one():
    global globvar    # Needed to modify global copy of globvar
    globvar = 1

def print_globvar():
    print(globvar)     # No need for global declaration to read value of globvar

set_globvar_to_one()
print_globvar()       # Prints 1


"""
Class Attribute vs. Instance Attribute

instance attribute - a Python variable belonging to one, and only one, object. This variable is only accessible in the scope of this object and it is defined inside 
                        the constructor function, __init__(self,..) of the class.
class attribute    - a Python variable that belongs to a class rather than a particular object. It is shared between all the objects of this class and it is defined 
                        outside the constructor function, __init__(self,...), of the class.
"""
class ExampleClass(object):
    class_attribute = 0

    def __init__(self, instance_attribute):
        self.instance_attribute = instance_attribute


"""
Typically, you have at least two ways to manage an attribute. Either you can access and mutate the attribute directly or you can use methods. 
Methods are functions attached to a given class. They provide the behaviors and actions that an object can perform with its internal data and attributes.
If you expose your attributes to the user, then they become part of the public API of your classes. Your user will access and mutate them directly in their code. 
The problem comes when you need to change the internal implementation of a given attribute. Programming languages such as Java and C++ encourage you to 
never expose your attributes to avoid this kind of problem. Instead, you should provide getter and setter methods, also known as accessors and mutators, respectively. 
These methods offer a way to change the internal implementation of your attributes without changing your public API.

The main advantage of Python properties is that they allow you to expose your attributes as part of your public API. 
If you ever need to change the underlying implementation, then you can turn the attribute into a property at any time without much pain.
"""


"""
Python中的多态及抽象类: 多态(polymorphism), 大致可以理解为: 即使你不知道变量指向哪种形态, 也能够对其执行操作, 而且操作的行为将随对象的类型不同而不同。Python 默认就是多态的。
    - 祖先类 (Shape): 可以是抽象类 (没有 init 函数)，这意味着系统不允许你创建类的对象，抽象类什么具体的工作也不做，只是描述了他的全部后代的模样
    - 祖先类里定义的方法 (draw 和 getSize) 是后代必须至少要实现的, 这种描述是强制性的, 后代类里除了必须要实现祖先类里所有的方法之外，可以额外定义自己专属的方法（函数）
    - Shape这个抽象类存在的唯一意义就是规定了它的后代类的某些特征: 必须实现 draw()及getSize()方法。
    - 当且仅当Shape的某个后代类定义实现了Shape的全部抽象函数, 该类才可以被实例化 - 即允许创建该类型的对象。Triangle实现了Shape的全部抽象方法, 都不再“抽象”, 可以实例化
"""
# Shapes.py
from abc import ABC, abstractmethod
class Shape(ABC):
    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def getSize(self):
        pass


class Triangle(Shape):      # 后代1
    def __init__(self):
        self.point0 = (0,0)
        self.point1 = (0,0)
        self.point2 = (0,0)

    def draw(self):
        print("Triangle::draw")

    def getSize(self):
        pass       #detail omitted

    def getArea(self):
        return 0   #it should be w * h / 2


class Circle(Shape):        # 后代2
    def __init__(self):
        self.ptCenter = (0,0)
        self.iRadius = 0

    def draw(self):
        print("Circle::draw")

    def getSize(self):
        pass

    def getArea(self):
        return 0


"""
当一个页面被显示出来时, 软件会遍历这个列表, 然后逐一调用列表内Shape子对象的draw()方法, 以便把每个界面元素画出来。对你听得没错, 就是每个对象自己画自己。
因为三角形类了解三角形的数据表达形式, 掌握描绘一个三角形的全部信息, 由这个类的draw()来承担这个职责再合适不过了。圆形类也是一样。我们设想一下，假设在页面上描绘三角形的任务
不是由三角形类来完成，而是由外部代码来完成，那么外部代码就必须清楚并访问三角形对象内部的全部细节，如果这件事情真的发生的话，对于软件工程而言是灾难性的：外部代码知道太多关于三角形内部实现的细节！ 
内部实现的细节变成了接口的一部分！ 三角形类接口不再简洁明了！ 以后你如果想修改三角形的内部数据结构，这几乎不可能，因为外部代码也要跟着改，涉及的外部程序和修改点可能太多 --- 这种复杂的情况，
我们称之为紧耦合 - tight coupling。而程序的松散耦合 - loose coupling, 才是我们的目标。下面代码我们看到, renderDocument()函数并不清楚变量x的具体类型, 它只认为x是一个Shape, 实现了draw()方法,
至于x到底是三角形、圆形, 完全不关心。但是我们发现, x是什么类型, 就会执行什么类型的对应的draw()函数，并打印出对应的文字。这就是多态，这些变量类型未知，但自动展现出与类型对应的恰当行为。
"""

t1 = Triangle()
t2 = Triangle()
c1 = Circle()
c2 = Circle()

#doc模拟一个文档，将界面元素组织在列表中
doc = [c2,t2,c1,t1]

#遍历全部界面元素，将它们全部画出来
def renderDocument(doc):
    for x in doc:
        x.draw()
renderDocument(doc)



"""
Multi-Processing  vs  Multi-Threading       
什么是进程(process)和线程(thread):  进程是操作系统分配资源的最小单元, 线程是操作系统调度的最小单元。一个应用程序至少包括1个进程, 而1个进程包括1个或多个线程, 线程的尺度更小。
                                   每个进程在执行过程中拥有独立的内存单元，而一个线程的多个线程在执行过程中共享内存。

Python多进程和多线程哪个快?
由于GIL(python解释器中存在GIL(全局解释器锁), 它的作用就是保证同一时刻只有一个线程可以执行代码。)的存在, 很多人认为Python多进程编程更快, 
针对多核CPU, 理论上来说也是采用多进程更能有效利用资源。网上很多人已做过比较，我直接告诉你结论吧。
  - 对CPU密集型代码(比如循环计算) - 多进程效率更高
  - 对IO密集型代码(比如文件操作，网络爬虫) - 多线程效率更高。
为什么是这样呢? 其实也不难理解。对于IO密集型操作, 大部分消耗时间其实是等待时间, 在等待时间中CPU是不需要工作的, 那你在此期间提供双CPU资源也是利用不上的. 
相反对于CPU密集型代码, 2个CPU干活肯定比一个CPU快很多。那么为什么多线程会对IO密集型代码有用呢? 这时因为python碰到等待会释放GIL供新的线程使用, 实现了线程间的切换.

Multithreading is a technique that allows for concurrent (simultaneous) execution of two or more parts of a program for maximum utilization of a CPU.
1. Programs are made up of processes and threads -->
    A program is an executable file like chrome.exe, A process is an executing instance of a program, Thread is the smallest executable unit of a process
2. Developers should make use of multithreading for a few reasons:
    Higher throughput / Responsive applications that give the illusion of multitasking / Efficient utilization of resources
3. Basic Concepts of Multithreading:
    Processes are what actually execute the program / Threads are sub-tasks of processes and if synchronized correctly can give the application performing everything at once
    Concurrency / Context Switching (technique where CPU time is shared across all running processes and is key for multitasking) / 
    Thread Pools (allow you to decouple task submission and execution, consists of homogenous worker threads, use queue to manage) / 
    Locking (synchronization technique, mutex is a lock) / Mutex (互斥锁 mutual exclusion, allows only a single thread to access a resource) / Semaphore 信号量
    Thread Safety (different threads can access the same resources without error behavior or producing unpredictable results like a race condition or a deadlock) /
4. Issues Involved with Multiple Threads
    Deadlock / Race conditions / Starvation / Livelock

消息队列(Message Queue)简写为MQ, 消息队列中间件是分布式系统中重要的组件。主要解决应用解耦, 异步消息, 流量削锋, 消息通迅等问题, 从而实现高性能,高可用,可伸缩和最终一致性的架构。

REST APIs - Application Programming Interface, The main feature of REST API is statelessness, it defines GUPD get/update/put/delete, Clients and servers exchange data using HTTP
"""


a = [1, 2, 3]
b = [2, 3, 4]
exec('a+b')



# From graph view, a <tree> can also be defined as a "directed acyclic graph" (有向无环图) which has N nodes and N-1 edges.


# How to fix SyntaxError: positional argument follows keyword argument?
def add_numbers(a, b, c):
    return a+b+c

result = add_numbers(10, 20, 30)            # Scenario 1 – Use only Positional Arguments.
result = add_numbers(a=10, b=20, c=30)      # Scenario 2 – Use only Keyword Arguments.
result = add_numbers(10, b=20, c=30)          # Scenario 3 – Use Positional arguments first, followed by Keyword Arguments.

def g(a, b, *args, kw_only=None):
    print(f'{a=}, {b=}, {args=}, {kw_only=}')

def force_keyword_argument():
    g(1, 2, 3, 4, kw_only=3)        # a=1, b=2, args=(3, 4), kw_only=3      anything comes after *args, must be a keyword argument, cannot be positional argument



# Python program showing
# how to style import statements (example)

import math
import os
  
# Third party imports
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
  
# Local application imports
from local_module import local_class
from local_package import local_function



# We need to find a way to automate batch testing RE policies in local, issue: downloading bulk jsons from S3 errored out. 

"""
Amazon API Gateway: A Single Entry-Point:
API Gateway allows for handling common API management tasks such as security, caching, throttling, and monitoring. While its primary objective is to 
provide that ***abstraction layer*** on top of your backend APIs and microservices, it can also allow backends to be simple web applications for 
web portal access or Amazon S3 buckets for providing access to static web content or documents.
"""


"""
parameter和argument的区别:
1. parameter是指函数定义中参数, 而argument指的是函数调用时的实际参数。
2. 简略描述为: parameter=形参(formal parameter), argument=实参(actual parameter)。
3. 在不很严格的情况下，现在二者可以混用, 一般用argument, 而parameter则比较少用。
While defining method, variables passed in the method are called parameters.当定义方法时，传递到方法中的变量称为参数.
While using those methods, values passed to those variables are called arguments. 当调用方法时，传给变量的值称为引数.(有时argument被翻译为 "引数")
"""




#####################################  Pandas  #####################################

""" Describing Data """
# 1. Statistics and Counts
import numpy as np
import pandas as pd
names = ['age', 'workclass', 'fnlwgt', 'education', 'educationnum', 'maritalstatus', 'occupation', 'relationship', 'race',
        'sex', 'capitalgain', 'capitalloss', 'hoursperweek', 'nativecountry', 'label']
train_df = pd.read_csv("adult.data", header=None, names=names)
print(train_df.head())
print(train_df.describe())
print(train_df.info())

# 2. Converting data types
train_df['numeric_column'] = pd.to_numeric(train_df['string_column'])  # to_datetime() / to_string()

# 3. Finding unique values
print(train_df['relationship'].unique())
print(train_df['relationship'].value_counts())                   
print(train_df.groupby('relationship')['label'].value_counts(normalize=True))  # Group by relationship and then get the value counts of label with normalization

# Convert the string label into a value of 1 when >= 50k and 0 otherwise
train_df['label_int'] = train_df.label.apply(lambda x: ">" in x)
print(train_df.corr())      # Calculate correlations



""" Reshaping the Data """
# Pivot the data frame to show by relationship, workclass (rows) and label (columns) the average hours per week.
print(pd.pivot_table(train_df, values='hoursperweek', index=['relationship','workclass'], columns=['label'], aggfunc=np.mean).round(2))

# Calculate the frequencies between label and relationship, Crosstab with normalized outputs
print(pd.crosstab(train_df['label'], train_df.relationship, normalize=True))


import pandas.util.testing as tm
# Create long dataframe
def unpivot(frame):
    N, K = frame.shape
    data = {
        'value': frame.values.ravel('F'),
        'variable': np.asarray(frame.columns).repeat(N),
        'date': np.tile(np.asarray(frame.index), K)
    }
    return pd.DataFrame(data, columns=['date', 'variable', 'value'])
df = unpivot(tm.makeTimeDataFrame())

# Convert long to wide format
df_pivot = df.pivot(index='date', columns='variable', values='value')
# Convert back wide to long format
print(df_pivot.unstack())


"""  Cleaning Data  """
pd_series = pd.Series([5, 10, np.nan, 15, 20, np.nan, 25, 50, np.nan])
pd_series = pd_series.fillna(pd_series.mean())
pd_series = pd_series.dropna()      # Drop rows with missing data
print(pd_series.isnull())           # Show which rows are missing
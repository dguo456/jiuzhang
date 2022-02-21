# 648 · Unique Word Abbreviation
"""
An abbreviation of a word follows the form <first letter><number><last letter>
Assume you have a dictionary and given a word, find whether its abbreviation is unique 
in the dictionary. A word's abbreviation is unique if no other word from the dictionary 
has the same abbreviation.

Input:  [ "deer", "door", "cake", "card" ]
isUnique("dear")
isUnique("cart")
Output: false   true

Explanation:
Dictionary's abbreviation is ["d2r", "d2r", "c2e", "c2d"].
"dear" 's abbreviation is "d2r" , in dictionary.
"cart" 's abbreviation is "c2t" , not in dictionary.
"""
# 值得注意的是本题有可能需要大量调用isUnique()方法，所以最好要对字典提前做预处理，使得isUnique方法可以在 O(1)的时间完成
class ValidWordAbbr:
    """
    @param: dictionary: a list of words
    """
    def __init__(self, dictionary):
        self.abbr_dict = {}
        for word in dictionary:
            abbr = self.word2Vec(word)
            if abbr not in self.abbr_dict:
                self.abbr_dict[abbr] = set()
            # else:
            self.abbr_dict[abbr].add(word)

    def word2Vec(self, word):
        if len(word) < 3:
            return word
        return word[0] + str(len(word[1:-1])) + word[-1]

    """
    @param: word: a string
    @return: true if its abbreviation is unique or false
    """
    def isUnique(self, word):
        abbr = self.word2Vec(word)

        if abbr not in self.abbr_dict:
            return True

        for word_in_dict in self.abbr_dict[abbr]:
            if word_in_dict != word:
                return False

        return True



# Your ValidWordAbbr object will be instantiated and called as such:
# obj = ValidWordAbbr(dictionary)
# param = obj.isUnique(word)
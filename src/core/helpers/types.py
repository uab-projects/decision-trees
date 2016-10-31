"""
Checks if the given argument is a valid integer number or not

@param  num  number as string to check if is valid integer
@return true if number can be parsed to int
"""
def isInteger(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

"""
Checks if the given argument is a valid natural nubmer or not. 0 is natural, of course

@param  num  number as string to check if is valid natural number
@return true if number can be parsed to int and is positive
"""
def isNatural(num):
    return isInteger(num) and int(num) >= 0

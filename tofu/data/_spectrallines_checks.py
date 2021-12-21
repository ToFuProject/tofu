



# #############################################################################
# #############################################################################
#               ddata - special case: roman to int (SpectralLines)
# #############################################################################


def roman2int(ss):
    """
    :type s: str
    :rtype: int

    source: https://www.tutorialspoint.com/roman-to-integer-in-python
    """
    roman = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
        'IV': 4,
        'IX': 9,
        'XL': 40,
        'XC': 90,
        'CD': 400,
        'CM': 900,
    }
    i = 0
    num = 0
    while i < len(ss):
        if i+1 < len(ss) and ss[i:i+2] in roman:
            num += roman[ss[i:i+2]]
            i += 2
        else:
            num += roman[ss[i]]
            i += 1
    return num


def int2roman(num):
    roman = {
        1000: "M",
        900: "CM",
        500: "D",
        400: "CD",
        100: "C",
        90: "XC",
        50: "L",
        40: "XL",
        10: "X",
        9: "IX",
        5: "V",
        4: "IV",
        1: "I",
    }

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])



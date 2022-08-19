import numpy

def take_up_ram_new(gigabytes : int) -> str :
    """This is a dummy function for documentation testing 

    Args:
        gigabytes (int): _description_ in

    Returns:
        str: _description_ out
    """

    n = gigabytes * 1024
    result = [numpy.random.bytes(1024*1024) for x in range(n)]
    print(len(result))

    return "ok"
from treevalue import mapping, FastTreeValue


def func(x):
    print('func is called:', x)
    return x * 2


if __name__ == '__main__':
    t = FastTreeValue({
        'a': 20, 'b': 45,
        'x': {f'c{i}': 100 + i for i in range(1000)}
    })

    # use delayed in mapping function
    t1 = mapping(t, func, delayed=True)
    print(t1.b)
    # func is called: 45
    # 90

    print(t1.x.c2)
    # func is called: 102
    # 204

    print(t1.x.c3)
    # func is called: 103
    # 206

    print(t1.x.c50)
    # func is called: 150
    # 300

    print(t1)
    # all rest funcs will be called here
    # <structure of t1>

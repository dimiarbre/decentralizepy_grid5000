
def parse(h):
    res =  h.split(":")
    return [int(e) for e in res]

def to_hours(l):
    res = [str(e) for e in l]
    res = ["0" + e if len(e)==1 else e for e in res ]
    return ":".join(res)


def add_times(h1,h2):
    h1_parsed = parse(h1)
    h2_parsed = parse(h2)

    assert(len(h1_parsed) == len(h2_parsed))

    res = []
    s = 0
    for i in range(len(h1_parsed)-1,-1,-1):
        s += h1_parsed[i] + h2_parsed[i]
        if i >0:
            res.append(s%60)
            s = s//60
        else:
            res.append(s)
    return to_hours(res[::-1])

def to_sec(walltime):
    l = parse(walltime)
    return l[-1] + 60*(l[-2] + 60 * l[-3])


if __name__ == "__main__":
    walltime = "9:59:00"
    additionnal_time = "00:10:00"

    sum = add_times(walltime,additionnal_time)
    print(sum)
    print(to_sec(sum))
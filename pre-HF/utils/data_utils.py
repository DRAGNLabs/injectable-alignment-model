class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        s = "Struct: \n"
        for key, value in self.__dict__.items():
            s += f"{key}: {value} \n"
        return s



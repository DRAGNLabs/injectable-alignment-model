class Struct:
    """
    Struct class used to convert a dictionary to an object

    Used to serialize the YAML configuration file into a class object,
    where the keys of the dictionary are converted to attributes of the object.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        s = "Struct: \n"
        for key, value in self.__dict__.items():
            s += f"{key}: {value} \n"
        return s

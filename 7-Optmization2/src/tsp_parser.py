import pandas as pd

def parse_tsp(file_path):

    with open(file_path) as f:
        lines = f.readlines() # list containing lines of file
        # Columns to store data
        columns = []
        # Dictionary to store file structure
        data = {
            "NAME": "",
            "TYPE": "",
            "COMMENT": "",
            "DIMENSION": "",
            "EDGE_WEIGHT_TYPE": ""
        }

        i = 1
        for line in lines:
            line = line.strip()
            match i:
                case 1:
                    data['NAME'] = line
                    i += 1
                case 2:
                    data['TYPE'] = line
                    i += 1
                case 3:
                    data['COMMENT'] = line
                    i += 1
                case 4:
                    data['DIMENSION'] = line
                    i += 1
                case 5:
                    data['EDGE_WEIGHT_TYPE'] = line
                    i += 1
                case 6:
                    i += 1
                case _:
                    if line == "EOF":
                        break
                    columns.append([item.strip() for item in line.split()])

        data["DATA"] = columns

    return data
from typing import List, Dict, NamedTuple


class Section(NamedTuple):
    stype: str
    options: Dict[str, str]


def parse_darknet_cfg(cfg_file: str) -> List[Section]:
    options = []
    with open(cfg_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            if line.startswith('['):
                current = Section(stype=line, options={})
                options.append(current)
            else:
                key, value = line.split('=')
                options[-1].options[key.strip()] = value.strip()
    return options

# print(parse_darknet_cfg('/home/a/PROJ/AlexeyAB/darknet/cfg/yolov1/tiny-yolo.cfg'))

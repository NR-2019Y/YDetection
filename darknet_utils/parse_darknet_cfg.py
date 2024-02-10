from typing import List, Dict, NamedTuple


class Section(NamedTuple):
    stype: str
    options: Dict[str, str]


def parse_darknet_cfg(cfg_file: str, net_section_only: bool = False) -> List[Section]:
    options = []
    with open(cfg_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            if line.startswith('['):
                if net_section_only and len(options):
                    assert options[0].stype == "[net]"
                    return options
                current = Section(stype=line, options={})
                options.append(current)
            else:
                key, value = line.split('=')
                options[-1].options[key.strip()] = value.strip()
    return options


class NetConfig(NamedTuple):
    input_height: int
    input_width: int
    input_channels: int


def get_net_config(cfg_file: str):
    net_section = parse_darknet_cfg(cfg_file, net_section_only=True)[0]
    return NetConfig(
        input_height=int(net_section.options['height']),
        input_width=int(net_section.options['width']),
        input_channels=int(net_section.options['channels'])
    )

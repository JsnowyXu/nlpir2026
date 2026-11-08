"""Conservative name-based organization screening, with explicit uncertainty.

This is a transparent heuristic, not validated legal entity resolution. Unknown
names are never asserted to be people. Full raw records remain in SQLite.
"""
import re

INSTITUTION = re.compile(r'大学|学院|学校|研究院|研究所|研究中心|实验室|医院|委员会|政府|科学院|工程院|'
    r'(?<!\w)(?:universit\w*|institute|institution|college|academy|hospital|school|'
    r'ministry|government|department|laborator\w*|research\s+council|research\s+organization)(?!\w)', re.I)
COMPANY = re.compile(r'公司|集团|工厂|制药厂|制造厂|机械厂|电子厂|化工厂|银行|合作社|事务所|企业|'
    r'(?<!\w)(?:ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|company|'
    r'gmbh|llc|llp|plc|pte\.?|pvt\.?|co\.?|sarl|sas|srl|aktiengesellschaft|'
    r'industries|technologies|electronics|pharmaceuticals|holdings)(?!\w)', re.I)


def entity_type(name):
    if INSTITUTION.search(name):
        return 'institution'
    if COMPANY.search(name):
        return 'company'
    return 'unresolved'


def eligible(name, scope='organizations'):
    category = entity_type(name)
    if scope == 'all_applicants':
        return True
    if scope == 'companies':
        return category == 'company'
    if scope == 'organizations':
        return category in ('company', 'institution')
    raise ValueError('Unknown entity scope: '+scope)

"""
股票工具模块
根据股票代码判断股票类型，并返回对应的涨停比例
"""

from typing import Dict


def determine_stock_type(stock_code: str) -> str:
    """
    根据股票代码判断股票类型

    Args:
        stock_code: 股票代码（6位数字字符串）

    Returns:
        股票类型：'st', 'gem', 'kcb', 'bse', 'normal'

    股票类型映射：
        - 'st': ST股票（5%涨停）
        - 'gem': 创业板（30xxxx，20%涨停）
        - 'kcb': 科创板（688xxxx，20%涨停）
        - 'bse': 北交所（8xxxx/4xxxx，30%涨停）
        - 'normal': 普通股（10%涨停）
    """
    # 确保是字符串
    stock_code = str(stock_code).zfill(6)

    # 判断ST股票（股票代码包含ST标记）
    # 注意：这里假设股票代码本身不含ST标记，需要从其他列判断
    # 在实际应用中，应该从股票名称或专门的标记列判断
    if stock_code.endswith('ST') or 'ST' in stock_code:
        return 'st'

    # 判断创业板（30xxxx）
    if stock_code.startswith('30'):
        return 'gem'

    # 判断科创板（688xxx, 689xxx）
    if stock_code.startswith(('688', '689')):
        return 'kcb'

    # 判断北交所（8xxxx, 4xxxx, 9xxxx）
    if stock_code.startswith(('8', '4', '9')):
        return 'bse'

    # 默认为普通股
    return 'normal'


def get_limit_ratio(stock_type: str) -> float:
    """
    根据股票类型返回涨停比例

    Args:
        stock_type: 股票类型（'st', 'gem', 'kcb', 'bse', 'normal'）

    Returns:
        涨停比例（小数形式，如0.10表示10%）

    Raises:
        ValueError: 未知的股票类型
    """
    limit_ratio_map: Dict[str, float] = {
        'st': 0.05,      # ST股票：5%涨停
        'gem': 0.20,     # 创业板：20%涨停
        'kcb': 0.20,     # 科创板：20%涨停
        'bse': 0.30,     # 北交所：30%涨停
        'normal': 0.10   # 普通股：10%涨停
    }

    if stock_type not in limit_ratio_map:
        raise ValueError(f"未知的股票类型: {stock_type}")

    return limit_ratio_map[stock_type]


def is_st_stock(stock_code: str, stock_name: str = None) -> bool:
    """
    判断是否为ST股票

    Args:
        stock_code: 股票代码
        stock_name: 股票名称（可选，用于更准确判断）

    Returns:
        是否为ST股票
    """
    # 如果有股票名称，检查名称中是否包含ST
    if stock_name:
        stock_name_upper = stock_name.upper()
        if 'ST' in stock_name_upper or '*ST' in stock_name_upper:
            return True

    # 检查股票代码中是否包含ST标记
    stock_code_str = str(stock_code)
    if 'ST' in stock_code_str.upper():
        return True

    return False


def get_stock_info(stock_code: str, stock_name: str = None) -> Dict[str, any]:
    """
    获取股票的完整信息

    Args:
        stock_code: 股票代码
        stock_name: 股票名称（可选）

    Returns:
        包含股票类型和涨停比例的字典
    """
    stock_code_str = str(stock_code).zfill(6)

    # 判断是否为ST股票
    is_st = is_st_stock(stock_code_str, stock_name)

    # 确定股票类型
    if is_st:
        stock_type = 'st'
    else:
        stock_type = determine_stock_type(stock_code_str)

    # 获取涨停比例
    limit_ratio = get_limit_ratio(stock_type)

    return {
        'stock_code': stock_code_str,
        'stock_type': stock_type,
        'limit_ratio': limit_ratio,
        'is_st': is_st
    }
nl = 'Is Solon Borland serving as a representative?'
if any(item in nl for item in ['how many', 
                                        'what is the number', 
                                        'what was the number',
                                        'what are the total number',
                                        'what was the total number',
                                        'what rank']):
    print('000')
    pass
elif any(item in nl for item in ['more/less', 'more or less']) and '1' in ['0','1']:
    replace_dict = {'0':'less', '1':'more'}
    print('111')
elif any(item in nl for item in ['above or below', 'above/below']) and '1' in ['0','1']:
    replace_dict = {'0':'below', '1':'above'}
    print('222')
elif any(nl.startswith(prefix) for prefix in ['is', 'was', 'does', 'do', 'did', 'were']) and '1' in ['0','1']:
    replace_dict = {'0':'no', '1':'yes'}
    print('333')
elif any(item in nl for item in ['month']) and '1' in [str(n) for n in range(1,13)]:
    replace_dict = {
        '1': 'January',
        '2': 'February',
        '3': 'March',
        '4': 'April',
        '5': 'May',
        '6': 'June',
        '7': 'July',
        '8': 'August',
        '9': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
    }
    print('444')


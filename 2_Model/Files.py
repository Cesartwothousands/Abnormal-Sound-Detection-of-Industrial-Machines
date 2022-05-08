machines = ['fan','pump','slider','valve']
kinds = ['normal', 'abnormal']
rootpath = f'F:/毕业论文/Pictures/Mel/'
def name_path(name):
    paths = []
    for machine in machines:
        paths.append(rootpath+f'{machine}/{name}')
    return paths
file_names = {'normal': name_path(kinds[0]),'abnormal': name_path(kinds[1])}
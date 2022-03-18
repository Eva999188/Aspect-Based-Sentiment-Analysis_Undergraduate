import xlrd
import xlwt

def import_excel(excel):
    tables = []
    for rown in range(excel.nrows):
        array = {'content': '', 'thesis': '', 'story': '', 'people': '', 'photo': '', 'effect':''}
        array['content'] = table.cell_value(rown, 0)
        array['thesis'] = table.cell_value(rown, 1)
        array['story'] = table.cell_value(rown, 2)
        array['people'] = table.cell_value(rown, 3)
        array['photo'] = table.cell_value(rown, 4)
        array['effect'] = table.cell_value(rown, 5)
        tables.append(array)
    return tables

if __name__ == '__main__':
    #读
    data = xlrd.open_workbook(r'F:\jindian\nlp\test.xlsx')
    table = data.sheets()[0]
    tables = import_excel(table)
    keys = list(tables[0].keys())
    print(keys)
    #写
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('My Worksheet')
    workbook.save(r'F:\jindian\nlp\test1.xlsx')

    for i in range(len(tables)):
        iswrite=[0,0,0,0,0,0]
        for j in range(len(tables[0])):
            if i == 0:
               print('write1:', 0, ',', j)
               worksheet.write(0, j, label=tables[0][keys[j]])
               iswrite[j] = 1

               # print('iswrite',j,':',iswrite)
               continue
            elif j == 0:
               print('write2:', i, ',', 0)
               worksheet.write(i, 0 , label=tables[i][keys[0]])
               iswrite[0] = 1

            else:
               if tables[i][keys[j]] != '':
                  temp = tables[i][keys[j]].split('#', 1)
                  for k in range(len(tables[0])):
                     if temp[0] == tables[0][keys[k]]:
                        print('write3:', i, ',', k)
                        worksheet.write(i, k, label=int(temp[1]))

                        iswrite[k] = 1
                        break
                  temp = []
        # print(iswrite)
        for j in range(len(iswrite)):
            if j == 0:
                continue
            else:
                if iswrite[j] == 0:
                    print('write4:', i, ',', j)
                    worksheet.write(i, j, label= -2)


    workbook.save(r'F:\jindian\nlp\test1.xlsx')





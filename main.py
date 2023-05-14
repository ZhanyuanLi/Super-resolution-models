from xml.dom.minidom import Document, parse
import os


class XMLHandler:
    """Read the configuration file"""

    def __init__(self, xml_path, lista):
        self.xml_path = xml_path
        self.lista = lista

    def construct_xml(self):
        doc = Document()  # 创建文档对象
        cameras = doc.createElement('cameras')  # 创建根元素
        doc.appendChild(cameras)

        # camera
        camera = doc.createElement('camera')
        cameras.appendChild(camera)

        # point
        for num in range(len(lista)):
            point = doc.createElement('point')
            point.setAttribute('ID', str(num+1).zfill(3))
            camera.appendChild(point)
            # name\code\url
            name = doc.createElement('name')
            name_text = doc.createTextNode('11111111111')
            name.appendChild(name_text)
            point.appendChild(name)
            code = doc.createElement('code')
            code_text = doc.createTextNode('2222222222222')
            code.appendChild(code_text)
            point.appendChild(code)
            url = doc.createElement('url')
            url_text = doc.createTextNode('11111111111')
            url.appendChild(url_text)
            point.appendChild(url)

        # params
        params = doc.createElement('params')
        cameras.appendChild(params)

        # log
        log = doc.createElement('log')
        log.setAttribute('on_off', "on")
        log.setAttribute('level', "info")
        log_text = doc.createTextNode('11111111111')
        log.appendChild(log_text)
        params.appendChild(log)

        # log
        multiprocessing_on_off = doc.createElement('multiprocessing_on_off')
        multiprocessing_on_off_text = doc.createTextNode('11111111111')
        multiprocessing_on_off.appendChild(multiprocessing_on_off_text)
        params.appendChild(multiprocessing_on_off)

        # 将DOM对象写入文件
        with open(self.xml_path, 'w') as f:
            doc.writexml(f, indent='', addindent='\t', newl='\n', encoding='UTF-8')

    def read_xml(self):
        """Parsing XML"""
        domTree = parse(self.xml_path)
        rootNode = domTree.documentElement  # Document root element

        # All cameras
        camera = rootNode.getElementsByTagName("camera")[0]
        points = camera.getElementsByTagName("point")
        points_list = []
        for point in points:
            point_dic = {}
            point_dic['subtask_id'] = point.getAttribute("ID")
            # name
            name = point.getElementsByTagName("name")[0]
            point_dic['name'] = name.childNodes[0].data
            # code
            code = point.getElementsByTagName("code")[0]
            point_dic['code'] = code.childNodes[0].data
            # url
            url = point.getElementsByTagName("url")[0]
            point_dic['url'] = url.childNodes[0].data

            points_list.append(point_dic)

        # All params
        params = rootNode.getElementsByTagName("params")[0]
        log_list = {}
        log = params.getElementsByTagName("log")[0]
        log_list['on_off'] = log.getAttribute("on_off")
        log_list['level'] = log.getAttribute("level")
        log_list['log_fullpath'] = log.childNodes[0].data

        multiprocessing_on_off = params.getElementsByTagName("multiprocessing_on_off")[0]
        multip_on_off = multiprocessing_on_off.childNodes[0].data

        return points_list, log_list, multip_on_off


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lista = [111111, 22222222, 3333333333, 44444444]
    x = XMLHandler(os.getcwd() + "/test.xml", lista)
    x.construct_xml()
    points_list, log_list, multip_on_off = x.read_xml()
    print(points_list)

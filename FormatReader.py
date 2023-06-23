import xml.etree.ElementTree as et


class FormatReader:

    def getFormatByName(self, name):
        file = et.parse('./content/world-cup-formats.xml')
        root = file.getroot()
        for forms in root.iter('format'):
            if forms.attrib['name'] == name:
                return forms

    def getTeamsGroups(self, name=""):
        forms = self.getFormatByName(name)
        return [int(forms.find('teams').text), int(forms.find('groups').text)]

    def getRounds(self, name=""):
        forms = self.getFormatByName(name)
        rounds = []
        for r in forms.find('rounds').findall('round'):
            rounds.append(r)
        return rounds

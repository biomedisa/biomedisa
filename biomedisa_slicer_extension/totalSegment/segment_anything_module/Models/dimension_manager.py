

class DimensionManager():
    def __init__(self, color):
        self.color = color

    def getDimensionIndices(self):
        if(self.color == 'Red'):
            xIndex = 0
            yIndex = 1
            zIndex = 2
        elif(self.color == 'Green'):
            raise NotImplemented("TODO")
            #TODO
        elif(self.color == 'Yellow'):
            raise NotImplemented("TODO")
            #TODO
        else:
            raise Exception(f"Invalid color '{self.color}'. ")
        
        return [xIndex, yIndex, zIndex]

    def getForegroundCoords(self, coords, index):
        indices = self.getDimensionIndices()
        filtered_list = [[item[indices[0]], item[indices[1]]] for item in coords if item[indices[2]] == index and item[3] == True]
        
        for item in filtered_list:
            print(f"x: {item[0]}, y: {item[1]}")

        return filtered_list
    
    def getBackgroundCoords(self, coords, index):
        indices = self.getDimensionIndices()
        filtered_list = [[item[indices[0]], item[indices[1]]] for item in coords if item[indices[2]] == index and item[3] == False]
        return filtered_list
    
    def getSlice(self, image, index):
        if(self.color == 'Red'):
            return image[:, :, index]
        elif(self.color == 'Green'):
            return image[:, index, :]
        elif(self.color == 'Yellow'):
            return image[index, :, :]
        
    def setSlice(self, image, index, slice):
        if(self.color == 'Red'):
            image[:, :, index] = slice
            pass
        elif(self.color == 'Green'):
            pass
        elif(self.color == 'Yellow'):
            pass
        
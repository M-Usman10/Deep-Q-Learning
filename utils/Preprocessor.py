from skimage.color import rgb2gray
import skimage.transform as transform
class Frame_Processor:
   def __init__(self,crop=True,convert_to_grayscale=True,normalize=True,coords={},resize=True,resize_y=None,resize_x=None):
       self.crop=crop
       self.top_left_x=coords.get('top_left_x',0)
       self.top_left_y=coords.get('top_left_y',0)
       self.bottom_right_x=coords.get('bottom_right_x',-1)
       self.bottom_right_y=coords.get('bottom_right_y',-1)
       self.convert_to_grayscale=convert_to_grayscale
       self.normalize=normalize
       self.resize=resize
       self.resize_x=resize_x
       self.resize_y=resize_y

   def process_frame(self,frame):
       if self.convert_to_grayscale:
           frame=rgb2gray(frame)
       if self.crop:
           frame=frame[self.top_left_y:self.bottom_right_y,self.top_left_x:self.bottom_right_x]
       if self.normalize:
           frame=frame/255.0
       if self.resize:
           frame=transform.resize(frame, [self.resize_y,self.resize_x])
       return frame
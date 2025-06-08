import cv2
import modelo as Modelo
import time

class ShopIA:
    
    
    def init(self):
        self.cap=self.inicap()
        return self.cap
    
    def inicap(self):
        cap=cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,720)
        return cap
    
    def tiendaIA(self,cap):
        color=(0,255,0)
        classengine = Modelo.Model("Modelos/yolo11x.engine",(640,640))
        
        prev_time = 0
        
        while True:
            start_time = time.time()
            
            ret,frame=cap.read()
            frame_raw,frame_procesed=classengine.preprocess(frame)
            results=classengine.resultss("debug_preprocessed.jpg")
            img = results[0].orig_img.copy()
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0]/640)
                cls_id = int(box.cls[0])
                conf = box.conf[0]
                label = results[0].names[cls_id]
                
                al,an,c=frame.shape
                x1,y1=int(x1*an), int(y1*al)
                x2,y2= int(x2*an), int(y2*al)
                
                frame= cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                text = f"{label} {conf:.2f}"
                frame=cv2.putText(frame, text, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            current_time = time.time()
            fps = 1 / (current_time - start_time)
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
            
            cv2.imshow("Tienda IA", frame)
            t = cv2.waitKey(5)
            if t == 27:
                break
            
        self.cap.release()
        cv2.destroyAllWindows()
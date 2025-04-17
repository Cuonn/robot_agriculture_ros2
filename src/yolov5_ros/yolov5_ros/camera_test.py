import cv2

def main():
    # Mở camera mặc định (thường là camera 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở camera!")
        return
    
    print("Nhấn 'q' để thoát...")
    
    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()
        
        if not ret:
            print("Không nhận được frame!")
            break
        
        # Hiển thị hình ảnh
        cv2.imshow('Camera Feed', frame)
        
        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
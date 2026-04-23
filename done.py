from picamera2 import Picamera2
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageTk
import sqlite3
import tkinter as tk
from tkinter import messagebox
import os

camera_open = False
keyboardProc=None
#setup models
mtcnn = MTCNN(image_size=160, keep_all=False, device='cpu', post_process=False)#raspberry pi doesnt have cuda cores
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')

#setup database
con=sqlite3.connect("faces.db")
cur=con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS persons(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT,relationship TEXT)")#only creates table if its not yet been made. Has a person id to identify person, their name and relation to patient. If this was in one table, the person could only have one embedding, allowing for more than one increases accuracy.
cur.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT,personID INTEGER,embedding BLOB,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,FOREIGN KEY (personID) REFERENCES persons(id))  ")#creation of another table, new id for embeddings, id is used as a foreign key to link the tables, being made as personID
con.commit()

def openKeyboard():
    global keyboardProc
    if keyboardProc is None:
        keyboardProc=os.popen("matchbox-keyboard &")

def closeKeyboard():
    os.system("pkill matchbox-keyboard")

def openCamera():
    print("Opening Camera")
    global camera_open
    
    if camera_open:
        print("Camera already open")
        return None
    camera_open=True
    
    picam2 = Picamera2()
    config=picam2.create_preview_configuration(main={"size": (480, 360)})
    picam2.configure(config)
    picam2.start()

    cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    import time
    time.sleep(1)

    captured={"frame":None}
    def onTouch(event, x, y, flags, param):
        if event==cv2EVENTLBUTTONDOWN:
            captured["frame"]=frame.copy

    while True:
        
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        print("Camera running")
        cv2.putText(frame, "Tap screen to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Camera", frame)
        
        if captured["frame"] is not None:
            picam2.stop()
            picam2.close()
            cv2.destroyAllWindows()
            camera_open=False
            return captured["frame"]
        
        if cv2.waitKey(1)==27: # esc key
            picam2.stop()
            picam2.close()
            cv2.destroyAllWindows()
            camera_open=False
            return None
        

    cv2.destroyAllWindows()
    
def mainMenu():
    root = tk.Tk()
    root.title("Main menu")
    root.configure(background="white")
    root.attributes("-fullscreen", True)
    root.update_idletasks()
    root.update()

    tk.Label(root, text="Face Recognition System", font=("Arial", 14)).pack(pady=10)
    tk.Button(root, text="Recognise Person", width=25, command=lambda: startCameraFlow("recognise", root)).pack(pady=10)
    tk.Button(root, text="Add New Person", width=25, command=lambda: startCameraFlow("add", root)).pack(pady=10)
    tk.Button(root, text="Exit", command =root.destroy).pack(pady=10)
    tk.Button(root, text="Clear Database", width=25, command=clearDB).pack(pady=10)
    root.mainloop()

def startCameraFlow(mode, root):
    print("Camera flow started:", mode)
    frame=openCamera()
    if frame is not None:
        processFrame(frame, mode, root)

def eucDist(a, b):
    return np.sqrt(np.sum((a-b)**2))

def processFrame(frame, mode, root):
    image=frame#relic of when used opencv for camera, and had to convert colors, no time to remove
    croppedImage=mtcnn(image)
    
    if croppedImage is None:
        messagebox.showinfo("No face", "No face detected. Try again")
        return
    
    embedding = resnet(croppedImage.unsqueeze(0)).detach().numpy()[0]
    
    if mode=="add":
        showAddPersonScreen(frame, embedding, mode, root)
        return
    #recognise mode

    name, relationship, dist = recognise(embedding)
    
    if dist < 0.7:
        showResult(frame, name, relationship, root)
    else:
        showAddPersonScreen(frame, name, relationship, root)
        
def addPerson(name, relationship):
    cur.execute("INSERT INTO persons (name, relationship) VALUES (?, ?)", (name, relationship))
    con.commit()
    return cur.lastrowid
        
def saveEmbedding(personID, embedding):
    embeddingBytes=embedding.astype(np.float32).tobytes()
    cur.execute("INSERT INTO embeddings (personID, embedding) VALUES (?, ?)", (personID, embeddingBytes))
    con.commit()
    
def recognise(embedding):
    cur.execute("SELECT embeddings.embedding, persons.name, persons.relationship FROM embeddings JOIN persons ON embeddings.personID = persons.id")
    rows=cur.fetchall()
    
    bestMatch=None
    bestRelationship = None
    bestDistance=float("inf")
    
    for row in rows:
        dbEmbedding=np.frombuffer(row[0], dtype=np.float32)
        name=row[1]
        relationship=row[2]
        
        dist=eucDist(embedding, dbEmbedding)
        
        if dist<bestDistance:
            bestDistance=dist
            bestMatch=name
            bestRelationship=relationship
    return bestMatch, bestRelationship, bestDistance
        
def tkColourConvert(frame):
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img)
    img=img.resize((200,200))
    return ImageTk.PhotoImage(img)
        
def showResult(frame, name, relationship, root):
    window = tk.Toplevel(root)
    window.title("Result")
    
    window.attributes("-fullscreen", True)
    window.transient(root)
    window.grab_set()
    window.focus_set()

    img=tkColourConvert(frame)
    
    labelImg=tk.Label(window, image=img)
    labelImg.image=img
    labelImg.pack()
    
    tk.Label(window, text=f"Recognised as {name}\nRelationship: {relationship}", font=("Arial", 14)).pack(pady=10)
    tk.Button(window, text="Back", command=lambda:[window.grab_release(), window.destroy()]).pack(pady=10)
    
def clearDB():
    confirm = messagebox.askyesno("Confirm", "Delete all data?")
    
    if confirm:
        cur.execute("DELETE FROM embeddings")
        cur.execute("DELETE FROM persons")
        cur.execute("DELETE FROM sqlite_sequence WHERE name='persons'")
        cur.execute("DELETE FROM sqlite_sequence WHERE name='embeddings'")
        con.commit()
        messagebox.showinfo("Done", "Database cleared")

def showAddPersonScreen(frame, embedding, mode, root):
    window = tk.Toplevel(root)
    window.title("Add Person")
    window.attributes("-fullscreen", True)
    window.transient(root)  
    window.grab_set()   
    window.focus_set()


    img=tkColourConvert(frame)

    labelImg = tk.Label(window, image=img, bg="white")
    labelImg.image = img
    labelImg.grid(row=0, column=0, rowspan=5, padx=10, pady=10)

    tk.Label(window, text="Is this the correct photo?", bg="white",
             font=("Arial", 12)).grid(row=0, column=1, pady=5)

    tk.Label(window, text="Name:", bg="white").grid(row=1, column=1)
    nameEntry = tk.Entry(window)
    nameEntry.grid(row=2, column=1, pady=5)

    tk.Label(window, text="Relationship:", bg="white").grid(row=3, column=1)
    relEntry = tk.Entry(window)
    relEntry.grid(row=4, column=1, pady=5)

    nameEntry.focus_set()

    nameEntry.bind("<Button-1>", lambda e: [nameEntry.focus_set(), open_keyboard()])
    relEntry.bind("<Button-1>", lambda e: [relEntry.focus_set(), open_keyboard()])

    def savePerson():
        closeKeyboard()
        name = nameEntry.get()
        rel = relEntry.get()

        if not name.strip():
            messagebox.showwarning("Error", "Enter a name")
            return

        pid = addPerson(name, rel)
        saveEmbedding(pid, embedding)
        messagebox.showinfo("Saved", "Person saved succesfully.")
        window.grab_release()
        window.destroy()

    def retry():
        closeKeyboard
        window.grab_release()
        window.destroy()
        startCameraFlow(mode, root)

    # buttons on tkinter screen
    tk.Button(window, text="Save", command=savePerson, height=2, width=10).grid(row=5, column=1, pady=5)
    tk.Button(window, text="Retry", command=retry).grid(row=6, column=1, pady=5)
    tk.Button(window, text="Back", command=lambda: [closeKeyboard(), window.destroy()]).grid(row=7, column=1, pady=5)
mainMenu()
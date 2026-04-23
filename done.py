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

def createKeyboard(parent, entry):
    keys = [
        ['Q','W','E','R','T','Y','U','I','O','P'],
        ['A','S','D','F','G','H','J','K','L'],
        ['Z','X','C','V','B','N','M'],
        ['SPACE','BACK']
    ]

    frame = tk.Frame(parent, bg="black")
    frame.pack(expand=True, fill="both")

    def press(k):
        if k == "SPACE":
            entry.insert(tk.END, " ")
        elif k == "BACK":
            if len(entry.get()) > 0:
                entry.delete(len(entry.get())-1)
        else:
            entry.insert(tk.END, k)

    for row in keys:
        r = tk.Frame(frame, bg="black")
        r.pack(expand=True)

        for k in row:
            if k == "SPACE":
                w = 10
            elif k == "BACK":
                w = 8
            else:
                w = 4

            tk.Button(
                r,
                text=k,
                font=("Arial", 14),
                width=w,
                height=2,
                command=lambda x=k: press(x)
            ).pack(side="left", expand=True, padx=2, pady=2)

    return frame

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
        if event==cv2.EVENT_LBUTTONDOWN:
            captured["frame"]=frame.copy()
    cv2.setMouseCallback("Camera", onTouch)

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
    tk.Label(root, text="Face Recognition System", font=("Arial", 28)).pack(pady=30)

    tk.Button(root, text="Recognise Person", font=("Arial", 20), width=20, height=2,
              command=lambda: startCameraFlow("recognise", root)).pack(pady=20)

    tk.Button(root, text="Add New Person", font=("Arial", 20), width=20, height=2,
              command=lambda: startCameraFlow("add", root)).pack(pady=20)

    tk.Button(root, text="Clear Database", font=("Arial", 18), width=18, height=2,
              command=clearDB).pack(pady=20)

    tk.Button(root, text="Exit", font=("Arial", 18), width=12, height=2,
              command=root.destroy).pack(pady=20)
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
        showAddPersonScreen(frame, embedding, mode, root)
        
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
    window.geometry("1024x600")
    window.lift()
    window.focus_force()

    window.transient(root)
    window.grab_set()

    img = tkColourConvert(frame)

    labelImg = tk.Label(window, image=img)
    labelImg.image = img
    labelImg.pack(pady=20)

    tk.Label(
        window,
        text=f"Recognised as {name}\nRelationship: {relationship}",
        font=("Arial", 24)
    ).pack(pady=20)

    tk.Button(
        window,
        text="Back",
        font=("Arial", 20),
        width=10,
        height=2,
        command=lambda: [window.grab_release(), window.destroy()]
    ).pack(pady=20)
    
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

    # ===== TRUE FULLSCREEN =====
    window.overrideredirect(True)
    window.geometry(f"{window.winfo_screenwidth()}x{window.winfo_screenheight()}+0+0")
    window.lift()
    window.focus_force()
    window.grab_set()

    # ===== MAIN LAYOUT =====
    window.grid_rowconfigure(0, weight=3)  # content
    window.grid_rowconfigure(1, weight=1)  # keyboard
    window.grid_columnconfigure(0, weight=1)

    # ===== TOP AREA =====
    content = tk.Frame(window, bg="white")
    content.grid(row=0, column=0, sticky="nsew")

    content.grid_columnconfigure(0, weight=1)
    content.grid_columnconfigure(1, weight=1)

    # ===== IMAGE =====
    img = tkColourConvert(frame)
    imgLabel = tk.Label(content, image=img, bg="white")
    imgLabel.image = img
    imgLabel.grid(row=0, column=0, sticky="n", padx=40, pady=40)

    # ===== RIGHT SIDE =====
    right = tk.Frame(content, bg="white")
    right.grid(row=0, column=1, sticky="n", padx=40, pady=40)

    tk.Label(right, text="Is this the correct photo?",
             font=("Arial", 22), bg="white").pack(pady=20)

    tk.Label(right, text="Name:", font=("Arial", 18), bg="white").pack()
    nameEntry = tk.Entry(right, font=("Arial", 24), width=15)
    nameEntry.pack(pady=10)

    tk.Label(right, text="Relationship:", font=("Arial", 18), bg="white").pack()
    relEntry = tk.Entry(right, font=("Arial", 24), width=15)
    relEntry.pack(pady=10)

    # ===== BUTTONS =====
    btns = tk.Frame(right, bg="white")
    btns.pack(pady=20)

    # ===== KEYBOARD AREA (FIXED SPACE) =====
    keyboardArea = tk.Frame(window, bg="black")
    # DO NOT grid it yet

    keyboardFrame = None

    def showKeyboard(entry):
        nonlocal keyboardFrame

        # show keyboard area ONLY when needed
        keyboardArea.grid(row=1, column=0, sticky="nsew")

        if keyboardFrame:
            keyboardFrame.destroy()

    keyboardFrame = createKeyboard(keyboardArea, entry)

    nameEntry.bind("<Button-1>", lambda e: showKeyboard(nameEntry))
    relEntry.bind("<Button-1>", lambda e: showKeyboard(relEntry))

    # ===== ACTIONS =====
    def savePerson():
        name = nameEntry.get()
        rel = relEntry.get()

        if not name.strip():
            messagebox.showwarning("Error", "Enter a name")
            return

        pid = addPerson(name, rel)
        saveEmbedding(pid, embedding)

        window.grab_release()
        keyboardArea.grid_forget()
        window.destroy()

    def retry():
        window.grab_release()
        keyboardArea.grid_forget()
        window.destroy()
        startCameraFlow(mode, root)

    tk.Button(btns, text="Save", font=("Arial", 20),
              width=12, height=2, command=savePerson).pack(pady=5)

    tk.Button(btns, text="Retry", font=("Arial", 18),
              width=12, height=2, command=retry).pack(pady=5)

    tk.Button(btns, text="Back", font=("Arial", 18),
              width=12, height=2,
              command=lambda: [window.grab_release(), window.destroy()]
              ).pack(pady=5)
mainMenu()
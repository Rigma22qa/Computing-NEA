import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageTk
import sqlite3
import tkinter as tk
from tkinter import messagebox


#setup models
mtcnn = MTCNN(image_size=160, keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

#setup database
con=sqlite3.connect("faces.db")
cur=con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS persons(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT,relationship TEXT)")#only creates table if its not yet been made. Has a person id to identify person, their name and relation to patient. If this was in one table, the person could only have one embedding, allowing for more than one increases accuracy.
cur.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT,personID INTEGER,embedding BLOB,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,FOREIGN KEY (personID) REFERENCES persons(id))  ")#creation of another table, new id for embeddings, id is used as a foreign key to link the tables, being made as personID
con.commit()

def openCamera():
    cam=cv2.VideoCapture(0)

    while cam.isOpened():
        ret,frame=cam.read()
        if not ret:#if the frame isnt returned, break.
            break
        
        cv2.putText(frame, "Press C to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, "Press ESC to exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Camera", frame)
        
        key = cv2.waitKey(1)
        if key==ord('c'):
            capturedFrame = frame.copy()
            cam.release()
            cv2.destroyAllWindows()
            return capturedFrame
        
        if key==27: # esc key
            return None
        
    cam.release()
    cv2.destroyAllWindows()
    
def mainMenu():
    root=tk.Tk()
    root.title("Main menu")
    root.configure(background="white")
    root.geometry("300x300")#shows width, length when window created.
    root.attributes("-topmost", True)
    tk.Label(root, text="Face Recognition System", font=("Arial", 14)).pack(pady=10)
    tk.Button(root, text="Recognise Person", width=25, command=lambda: startCameraFlow("recognise")).pack(pady=10)
    tk.Button(root, text="Add New Person", width=25, command=lambda: startCameraFlow("add")).pack(pady=10)
    tk.Button(root, text="Exit", command =root.destroy).pack(pady=10)
    tk.Button(root, text="Clear Database", width=25, command=clearDB).pack(pady=10)
    root.mainloop()

def startCameraFlow(mode):
    frame=openCamera()
    if frame is not None:
        processFrame(frame, mode)

def eucDist(a, b):
    return np.sqrt(np.sum((a-b)**2))

def processFrame(frame, mode):
    image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    croppedImage=mtcnn(image)
    
    if croppedImage is None:
        messagebox.showinfo("No face", "No face detected. Try again")
        return None #if face is not found return nothing to be displayed on the tkinter screen
    
    embedding = resnet(croppedImage.unsqueeze(0)).detach().numpy()[0]
    
    if mode=="add":
        showAddPersonScreen(frame, embedding, mode)
        return
    #recognise mode

    name, relationship, dist = recognise(embedding)
    
    if dist < 0.7:
        showResult(frame, name, relationship)
    else:
        showAddPersonScreen(frame, embedding, mode)
        
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
        
def showResult(frame, name, relationship):
    window = tk.Toplevel()
    window.title("Result")
    
    img=tkColourConvert(frame)
    
    labelImg=tk.Label(window, image=img)
    labelImg.image=img
    labelImg.pack()
    
    tk.Label(window, text=f"Recognised as {name}\nRelationship: {relationship}", font=("Arial", 14)).pack(pady=10)
    tk.Button(window, text="Back", command=lambda: [window.destroy(), startCameraFlow("recognise")]).pack(pady=10)
    
def clearDB():
    confirm = messagebox.askyesno("Confirm", "Delete all data?")
    
    if confirm:
        cur.execute("DELETE FROM embeddings")
        cur.execute("DELETE FROM persons")
        cur.execute("DELETE FROM sqlite_sequence WHERE name='persons'")
        cur.execute("DELETE FROM sqlite_sequence WHERE name='embeddings'")
        con.commit()
        messagebox.showinfo("Done", "Database cleared")

def showAddPersonScreen(frame, embedding, mode):
    window = tk.Toplevel()
    window.title("Add Person")
    window.configure(bg="white")

    img = tkColourConvert(frame)

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

    def savePerson():
        name = nameEntry.get()
        rel = relEntry.get()

        if not name.strip():
            messagebox.showwarning("Error", "Enter a name")
            return

        pid = addPerson(name, rel)
        saveEmbedding(pid, embedding)
        window.destroy()

    def retry():
        window.destroy()
        startCameraFlow(mode)

    # buttons on tkinter screen
    tk.Button(window, text="Save", command=savePerson).grid(row=5, column=1, pady=5)
    tk.Button(window, text="Retry", command=retry).grid(row=6, column=1, pady=5)
    tk.Button(window, text="Back", command=window.destroy).grid(row=7, column=1, pady=5)
mainMenu()
import tkinter as tk
import hw3 as main_script
 

root = tk.Tk()
root.title("Search Engine")
token_var = tk.StringVar()


root_label = tk.Label(root, text = 'Token:', font=('calibre',10, 'bold'))
token_entry = tk.Entry(root,textvariable = token_var, font=('calibre',10,'normal'))
#the number  of documents, the number of [unique] words, and the total size (in KB) of your index on disk. 
display_query=tk.Text(font=("calibre, 12"))
def search():
    token = token_var.get().lower()
    print("{} is the token being searched".format(token))
    
    #token will be used to run the main script 
    query_result = main_script.displayQueryResult(token)
    #display_query.config(text=query_result)
    display_query.config(state=tk.NORMAL)
    display_query.delete('1.0', tk.END)
    display_query.insert(tk.END,query_result)
    display_query.config(state=tk.DISABLED)

search_btn=tk.Button(root,text = 'Search', command = search)

#root.geometry('270x410')
root.resizable(False, False) 

root_label.grid(row=0,column=0)
token_entry.grid(row=0, column=1)
search_btn.grid(row=0,column=2)

display_query.grid(row=1,column=0,columnspan=3)
root.mainloop()
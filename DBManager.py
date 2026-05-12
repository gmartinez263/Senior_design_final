import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("URL")
key = os.environ.get("KEY")
supabase = create_client(url, key)

def get_all_cannabinoids():
    response = supabase.table("Cannabinoids").select("*").order("name").execute()
    return response.data

def get_all_terpenes():
    response = supabase.table("Terpenes").select("*").order("name").execute()
    return response.data

def get_all_tags():
    response = supabase.table("Tags").select("*").order("effect").execute()
    return response.data

def get_user_sessions(user_id):
    sessions_response = supabase.table("Sessions").select("*, Products(*)").eq("UID", user_id).order("id", desc=True).execute()
    sessions = sessions_response.data
    
    if not sessions:
        return []

    #Get all product IDs from sessions to fetch cannabinoids and terpenes in bulk
    product_ids = list(set(s['product_id'] for s in sessions))
    
    #Fetch product cannabinoids with names
    pc_response = supabase.table("ProductCannabinoids").select("*, Cannabinoids(name)").in_("product_id", product_ids).execute()
    product_cannabinoids = {}

    for pc in pc_response.data:
        pid = pc['product_id']

        if pid not in product_cannabinoids:
            product_cannabinoids[pid] = []
        
        pc['name'] = pc['Cannabinoids']['name']
        product_cannabinoids[pid].append(pc)
        
    #Fetch product terpenes with names
    pt_response = supabase.table("ProductTerpenes").select("*, Terpenes(name)").in_("Product_id", product_ids).execute()
    product_terpenes = {}

    for pt in pt_response.data:
        pid = pt['Product_id']

        if pid not in product_terpenes:
            product_terpenes[pid] = []
        
        pt['name'] = pt['Terpenes']['name']
        product_terpenes[pid].append(pt)
        
    #Fetch session tags
    session_ids = [s['id'] for s in sessions]
    st_response = supabase.table("SessionTags").select("*, Tags(*)").in_("session_id", session_ids).execute()
    session_tags = {}

    for st in st_response.data:
        sid = st['session_id']

        if sid not in session_tags:
            session_tags[sid] = []
            
        session_tags[sid].append(st['Tags']['effect'])

    #Combine data
    for session in sessions:
        pid = session['product_id']
        sid = session['id']
        
        session['cannabinoids'] = product_cannabinoids.get(pid, [])
        session['terpenes'] = product_terpenes.get(pid, [])
        session['effects'] = session_tags.get(sid, [])
        
    return sessions

def product_composition(product_id):
    product_response = supabase.table("Products").select("*").eq("id", product_id).single().execute()
    product = product_response.data
    
    if not product:
        return None
        
    pc_response = supabase.table("ProductCannabinoids").select("*").eq("product_id", product_id).execute()
    pt_response = supabase.table("ProductTerpenes").select("*").eq("Product_id", product_id).execute()
    
    product['cannabinoids'] = pc_response.data
    product['terpenes'] = pt_response.data
    
    return product

def add_product(name, brand, productType, notes, cannabinoids, terpenes):
    product_data = {
        "name": name,
        "brand": brand,
        "type": productType,
        "notes": notes
    }

    response = supabase.table("Products").insert(product_data).execute()
    product_id = response.data[0]['id']

    #Insert cannabinoids
    if cannabinoids:
        pc_data = []

        for c in cannabinoids:
            pc_data.append({
                "product_id": product_id,
                "cannabinoid_id": c['id'],
                "serving": c['serving']
            })
        
        supabase.table("ProductCannabinoids").insert(pc_data).execute()

    #Insert terpenes
    if terpenes:
        pt_data = []

        for tid in terpenes:
            pt_data.append({
                "Product_id": product_id,
                "Terpene_id": tid
            })

        supabase.table("ProductTerpenes").insert(pt_data).execute()

    return product_id

def add_cannabinoid(name):
    cannabinoid_data = {
        "name": name
    }
    response = supabase.table("Cannabinoids").insert(cannabinoid_data).execute()
    return response.data[0]['id']

def add_terpene(name):
    terpene_data = {
        "name": name
    }
    response = supabase.table("Terpenes").insert(terpene_data).execute()
    return response.data[0]['id']

def add_tag(effect_name):
    tag_data = {
        "effect": effect_name
    }
    response = supabase.table("Tags").insert(tag_data).execute()
    return response.data[0]['id']

def add_session(user_id, product_id, servings, notes, tag_ids):
    session_data = {
        "UID": user_id,
        "product_id": product_id,
        "servings": servings,
        "notes": notes
    }
    response = supabase.table("Sessions").insert(session_data).execute()
    session_id = response.data[0]['id']

    if tag_ids:
        st_data = []

        for t_id in tag_ids:
            st_data.append({
                "session_id": session_id,
                "tag_id": t_id
            })

        supabase.table("SessionTags").insert(st_data).execute()

    return session_id
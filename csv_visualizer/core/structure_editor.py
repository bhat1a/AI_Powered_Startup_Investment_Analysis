#from csv_visualizer.core.block_extractor import detect_target_block, extract_block
#from csv_visualizer.core.css_extractor import detect_css_selector, extract_css_block
#from csv_visualizer.core.html_extractor import detect_html_region, extract_html_block
#from csv_visualizer.core.patch_llm import ask_llm_edit
#from csv_visualizer.core.edit_router import choose_edit_target


def edit_files(query, files):

    target = choose_edit_target(query)

    if target == "js":
        block_name = detect_target_block(query)
        old = extract_block(files["script.js"], block_name)
        new = ask_llm_edit(query, block_name, old)
        files["script.js"] = files["script.js"].replace(old, new)

    elif target == "css":
        selector = detect_css_selector(query)
        old = extract_css_block(files["style.css"], selector)
        new = ask_llm_edit(query, selector, old)
        files["style.css"] = files["style.css"].replace(old, new)

    else:
        region = detect_html_region(query)
        old = extract_html_block(files["index.html"], region)
        new = ask_llm_edit(query, region, old)
        files["index.html"] = files["index.html"].replace(old, new)

    return files

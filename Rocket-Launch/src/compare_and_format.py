color_map = {
    # Black
    "input": "RGB(0, 0, 0)",  # This token was given as input
    # Blue
    "irm": "RGB(0, 102, 255)",  # The IRM outputs this token
    # Yellow
    "base": "RGB(204, 204, 0)", # The base model outputs this token
    # Green
    "same": "RGB(153, 255, 102)", # The base model and the IRM output the same token
    # Red
    "diff": "RGB(204, 51, 0)"  # Neither the base model nor the IRM output this token
}

fake_color_map = {
    "input": "input",
    "irm": "irm",
    "base": "base",
    "same": "same",
    "diff": "diff"
}

sample_irm_out = ["a", "b", "c", "d", "e", "f", "g"]
sample_base_out = ["a", "n", "c", "d", "l", "k", "g"]

sample_base_sequence = [
    ["a", "n", "o", "t", "h", "e", "g"],
    ["a", "b", "d"],
    ["a", "b", "c", "r"],
    ["a", "b", "c", "d", "e"],
    ["a", "b", "c", "d", "e", "k"],
    ["a", "b", "c", "d", "e", "f", "g"],
]

def compare_and_make_color_script(irm_out, base_sequence, base_out, prompt_len):

    # You can use other fonts, but this one is monospace
    font = "Consolas"
    size = 3.5

    tab = "    "
    # Define the prefixes for you VB variable lists
    word_list_pre = "word_list"
    color_list_pre = "color_list"
    ids = (a for a in range(100000000))

    # Formats the given sequence and colors as VB lists
    def to_macro(sequence, colors):
        s = ""
        id = next(ids)
        word_list = ""
        for word in sequence:
            word_list += f"\"{word}\", "
        word_list = word_list[:-2]

        color_list = ""
        for color in colors:
            color_list += f"{color}, "
        color_list = color_list[:-2]

        s += f"{tab}{word_list_pre}{id} = Array({word_list})\n{tab}{color_list_pre}{id} = Array({color_list})\n\n"

        return s

    source = base_out
    source_color = "base"
    compare = irm_out
    compare_color = "irm"

    color_sequence = []
    seq_step = 0

    # Define color assignments based on token comparison
    for seq in base_sequence:
        curr_colors = [color_map["input"]]
        for i in range(1, len(seq)):
            if i <= seq_step + prompt_len:
                curr_colors.append(color_map["input"])
            elif seq[i] == compare[i] and seq[i] == source[i]:
                curr_colors.append(color_map["same"])
            elif seq[i] != compare[i] and seq[i] != source[i]:
                curr_colors.append(color_map["diff"])
            elif seq[i] == compare[i] and seq[i] != source[i]:
                curr_colors.append(color_map["irm"])
            elif seq[i] != compare[i] and seq[i] == source[i]:
                curr_colors.append(color_map["base"])

        seq_step += 1
        color_sequence.append(curr_colors)

    # Define assignments for base and comparison full output lists
    all_macro = to_macro(irm_out, [color_map["input"]] * prompt_len + [color_map["irm"]] * (len(irm_out) - prompt_len))
    all_macro += to_macro(base_out, [color_map["input"]]* prompt_len + [color_map["base"]] *  (len(irm_out) - prompt_len))

    # Define assignments for subsequent comparison output lists
    for i in range(len(base_sequence)):
        all_macro += to_macro(base_sequence[i], color_sequence[i])

    num_prints = next(ids)

    # Prepare variable declarations
    all_dims = ""
    for i in range(num_prints):
        all_dims += f"{tab}Dim {word_list_pre}{i} as Variant\n"
        all_dims += f"{tab}Dim {color_list_pre}{i} as Variant\n\n"

    # Prepare function calls
    all_prints = ""
    for i in range(num_prints):
        all_prints += f"{tab}PrintWordsInColors {word_list_pre}{i}, {color_list_pre}{i}\n"

    # Output the VB Script
    print(
    f"""Sub PrintWordsInColors(word_list As Variant, color_list As Variant)
        Dim i As Integer
        Dim word As String
        Dim color As Long
        
        ' Set font to {font}
        Selection.Font.Name = "{font}"
        Selection.Font.Size = {size}
        
        ' Start a new line
        Selection.TypeText Chr(11)
        
        ' Loop through each word in word_list
        For i = LBound(word_list) To UBound(word_list)
            word = word_list(i)
            
            ' Get the corresponding color from color_list
            color = color_list(i)
            
            ' Set the font color
            Selection.Font.color = color
            
            ' Print the word
            Selection.TypeText word & " "
        Next i
    End Sub

    Sub Example()
    {all_dims}

    {all_macro}

    {all_prints}
    End Sub"""
    )

#compare_and_make_color_script(sample_irm_out, sample_base_sequence, sample_base_out)


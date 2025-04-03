 Fly Maze Tracker is an interactive video analysis tool designed to automate the detection and tracking of flies in light-based behavioral assays. This program enables researchers to study memory, preference, and movement in Drosophila (fruit flies) across illuminated zones, a method particularly useful in models of neurodegenerative diseases like Alzheimer’s.

 
 Features

    📁 Load and process any .mp4, .mov, or .avi video

    🎨 Prompt user to input experimental setup (top/bottom light color, number of flies)

    🎬 GUI video preview tool to crop start/end times before processing

    🧠 Automatic detection of fly crossings using background subtraction

    📊 Real-time frame processing with crossing counts

    📝 Export results to CSV with full metadata

 Installation

Make sure you are using Python 3.8 or later.
1. Install Dependencies

pip install opencv-python FreeSimpleGUI numpy

2. Clone or Download the Repository
 How to Run

python Detect_Fly_Maze_V7.py

 Usage Flow

    Select a video file from your system.

    Enter experiment setup in the GUI:

        Top light color

        Bottom light color

        Number of flies in each vial (Top/Bottom)

    Crop the video to the desired start and end time using a slider preview.

    Drag and confirm the line to set where fly crossings are detected (coming soon).

    The program starts processing the video:

        Frame-by-frame processing

        Fly crossings are detected across a horizontal line

        Real-time crossing count updates are shown

    Exit anytime with Ctrl+C — all data collected so far will be saved safely.

    Output CSV file is saved with:

        Top/bottom zone colors

        Fly counts baseline

        Timestamps of crossings

        Current zone counts after each crossing

📄 CSV Output Example

# Video: Movie-on-1-21-25-at-12.29-PM-2redB-7M-1GreenT.mov
# Start time: 2025-03-31 14:52:00
# Top color: Green
# Bottom color: Red
# Initial Top: 15, Bottom: 5

Frame,Timestamp(s),Direction,TopCount,BottomCount
1456,48.53,bottom_to_top,16,4
1490,49.66,top_to_bottom,15,5
...

🔮 Planned Features

Video cropping via slider

GUI-based line repositioning

Live zone count display

Export raw timestamps (optional)

    Enhanced debugging tools and background modeling


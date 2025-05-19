import streamlit.components.v1 as components

def custom_camera():
    custom_html = """
        <div style="position: relative;">
            <video id="video" width="400" height="300" autoplay></video>
            <canvas id="overlay" style="position: absolute; top: 0; left: 0;"></canvas>
        </div>
        <script>
            const video = document.getElementById('video');
            const overlay = document.getElementById('overlay');
            const ctx = overlay.getContext('2d');
            
            // Access webcam
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                    
                    // Draw guidelines
                    ctx.strokeStyle = 'red';
                    ctx.beginPath();
                    ctx.moveTo(200, 0);
                    ctx.lineTo(200, 300);
                    ctx.moveTo(0, 150);
                    ctx.lineTo(400, 150);
                    ctx.stroke();
                });
        </script>
    """
    components.html(custom_html, height=350)

# Use the custom component
custom_camera()
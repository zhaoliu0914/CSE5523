from PIL import Image

# Create a new image with size 2x2 pixels
img = Image.new('RGB', (2, 2))

# Set the pixel colors
img.putpixel((0, 0), (255, 0, 0))  # Red
img.putpixel((1, 0), (0, 255, 0))  # Green
img.putpixel((0, 1), (0, 0, 255))  # Blue
img.putpixel((1, 1), (255, 255, 0))  # Yellow

# Save the image
img.save('4_pixel_image.png')

# Display the image
img.show()

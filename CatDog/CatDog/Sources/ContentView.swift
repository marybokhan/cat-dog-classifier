import SwiftUI
import PhotosUI
import CoreML
import Vision

struct ContentView: View {
    
    @State private var showPhotoPicker = false
    @State private var selectedPhoto = [PhotosPickerItem]()
    @State private var selectedImage: UIImage?
    @State private var catProbability: Float = 0.0   // Raw model output for cat (index 0)
    @State private var dogProbability: Float = 0.0   // Raw model output for dog (index 1)
    @State private var isClassifying = false
    
    // Constants for progress indicator
    private let minValue: Float = 0.0
    private let maxValue: Float = 24.0
    
    let model = try? CatDogModel(configuration: MLModelConfiguration())

    var body: some View {
        VStack(spacing: 20) {
            if let selectedImage {
                Image(uiImage: selectedImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 250, height: 250)
                    .clipShape(RoundedRectangle(cornerRadius: 15))
                
                if isClassifying {
                    classifyingProgressView
                } else {
                    modelOutputView
                }
            }
            selectPhotoButton
        }
        .padding()
        .photosPicker(
            isPresented: $showPhotoPicker,
            selection: $selectedPhoto,
            maxSelectionCount: 1,
            matching: .images
        )
        .onChange(of: selectedPhoto) { 
            guard let item = selectedPhoto.first else { return }
            loadImage(from: item)
        }
    }
    
    private var classifyingProgressView: some View {
        ProgressView("Classifying...")
    }
    
    private var modelOutputView: some View {
        VStack {
            Text("Model Output:")
                .font(.headline)
                .padding(.bottom, 5)
            
            VStack(spacing: 10) {
                HStack {
                    Text("Cat:")
                        .frame(width: 80, alignment: .leading)
                    ProgressView(
                        value: abs(catProbability),
                        total: maxValue
                    )
                    Text(String(format: "%.2f", abs(catProbability)))
                        .frame(width: 60, alignment: .trailing)
                }
                
                HStack {
                    Text("Dog:")
                        .frame(width: 80, alignment: .leading)
                    ProgressView(
                        value: abs(dogProbability),
                        total: maxValue
                    )
                    Text(String(format: "%.2f", abs(dogProbability)))
                        .frame(width: 60, alignment: .trailing)
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding(.horizontal)
    }
    
    private var selectPhotoButton: some View {
        Button(action: {
            self.showPhotoPicker = true
        }) {
            Label("Select Photo", systemImage: "photo.fill")
        }
        .buttonStyle(.borderedProminent)
    }
    
    private func loadImage(from item: PhotosPickerItem) {
        Task {
            do {
                let data = try await item.loadTransferable(type: Data.self)
                if let data = data, let image = UIImage(data: data) {
                    await MainActor.run {
                        self.selectedImage = image
                        classifyImage(image)
                    }
                }
            } catch {
                print("Error loading image: \(error)")
            }
        }
    }
    
    private func classifyImage(_ image: UIImage) {
        guard let model = model else { return }
        
        isClassifying = true
        catProbability = 0.0
        dogProbability = 0.0
        
        Task {
            // Create an exact 224x224 pixel buffer for the model
            let targetSize = CGSize(width: 224, height: 224)
            guard let resizedImage = createExactSizeImage(image, targetSize: targetSize),
                  let buffer = resizedImage.toCVPixelBuffer() else {
                await MainActor.run {
                    isClassifying = false
                }
                return
            }
            
            do {
                let input = CatDogModelInput(input: buffer)
                let output = try await model.prediction(input: input)
                
                await MainActor.run {
                    let multiarray = output.var_358
                    
                    // Store the raw output values from the model
                    catProbability = multiarray[0].floatValue  // Index 0 represents cat
                    dogProbability = multiarray[1].floatValue  // Index 1 represents dog
                    
                    // Print raw values for debugging
                    print("Raw model output - Cat: \(catProbability), Dog: \(dogProbability)")
                    
                    isClassifying = false
                }
            } catch {
                print("Prediction error: \(error)")
                await MainActor.run {
                    isClassifying = false
                }
            }
        }
    }
}

extension ContentView {
    // Helper functions for image processing
    func resizeImage(_ image: UIImage, to targetSize: CGSize) -> UIImage? {
        let size = image.size
        
        let widthRatio = targetSize.width / size.width
        let heightRatio = targetSize.height / size.height
        
        let newSize: CGSize
        if widthRatio > heightRatio {
            newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
        } else {
            newSize = CGSize(width: size.width * widthRatio, height: size.height * widthRatio)
        }
        
        let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)
        
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: rect)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage
    }
    
    // Create an image with exact dimensions for ML model input
    func createExactSizeImage(_ image: UIImage, targetSize: CGSize) -> UIImage? {
        // Create a new context of exactly the target size
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        
        // First resize maintaining aspect ratio to cover the target size
        let size = image.size
        let widthRatio = targetSize.width / size.width
        let heightRatio = targetSize.height / size.height
        
        // Use the larger ratio to ensure the image covers the target area
        let ratio = max(widthRatio, heightRatio)
        let newSize = CGSize(width: size.width * ratio, height: size.height * ratio)
        
        // Calculate centering rect to keep the image centered
        let x = (targetSize.width - newSize.width) / 2
        let y = (targetSize.height - newSize.height) / 2
        let rect = CGRect(x: x, y: y, width: newSize.width, height: newSize.height)
        
        // Draw the image in the centered rect
        image.draw(in: rect)
        
        // Get the exact-sized image
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage
    }
}

#Preview {
    ContentView()
}

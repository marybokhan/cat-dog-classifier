import SwiftUI
import PhotosUI
import CoreML
import Vision

struct ContentView: View {
    
    @State private var showPhotoPicker = false
    @State private var selectedPhoto = [PhotosPickerItem]()
    @State private var selectedImage: UIImage?
    @State private var catProbability: Float = 0.0
    @State private var dogProbability: Float = 0.0
    @State private var isClassifying = false
    
    let model = try? CatDogModel(configuration: MLModelConfiguration())

    var body: some View {
        VStack(spacing: 20) {
            if let selectedImage {
                Image(uiImage: selectedImage)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 250, height: 250)
                    .clipShape(RoundedRectangle(cornerRadius: 15))
                
                if isClassifying {
                    ProgressView("Classifying...")
                } else {
                    VStack {
                        Text("Classification Results")
                            .font(.headline)
                            .padding(.bottom, 5)
                        
                        VStack(spacing: 10) {
                            HStack {
                                Text("Cat")
                                    .frame(width: 80, alignment: .leading)
                                ProgressView(value: Double(catProbability), total: 1.0)
                                Text(String(format: "%.1f%%", catProbability * 100))
                                    .frame(width: 60, alignment: .trailing)
                            }
                            
                            HStack {
                                Text("Dog")
                                    .frame(width: 80, alignment: .leading)
                                ProgressView(value: Double(dogProbability), total: 1.0)
                                Text(String(format: "%.1f%%", dogProbability * 100))
                                    .frame(width: 60, alignment: .trailing)
                            }
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .padding(.horizontal)
                }
            } else {
                Button(action: {
                    self.showPhotoPicker = true
                }) {
                    self.addPhotoButton
                }
            }
            
            Button(action: {
                self.showPhotoPicker = true
            }) {
                Label("Select Another Photo", systemImage: "photo.fill")
            }
            .buttonStyle(.borderedProminent)
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
    
    private var addPhotoButton: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 15, style: .continuous)
                .frame(width: 250, height: 250)
                .foregroundStyle(.blue)
            
            Image(systemName: "plus")
                .font(.largeTitle)
                .foregroundStyle(.white)
        }
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
            guard let resizedImage = resizeImage(image, to: CGSize(width: 224, height: 224)),
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
                    // Apply softmax to convert logits to probabilities
                    let catLogit = multiarray[0].floatValue
                    let dogLogit = multiarray[1].floatValue
                    
                    // Calculate softmax: exp(logit) / sum(exp(logits))
                    let catExp = exp(catLogit)
                    let dogExp = exp(dogLogit)
                    let sum = catExp + dogExp
                    
                    catProbability = catExp / sum
                    dogProbability = dogExp / sum
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

extension UIImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(self.size.width),
                                         Int(self.size.height),
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)
        
        guard status == kCVReturnSuccess else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                     width: Int(self.size.width),
                                     height: Int(self.size.height),
                                     bitsPerComponent: 8,
                                     bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
                                     space: rgbColorSpace,
                                     bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            return nil
        }
        
        context.translateBy(x: 0, y: self.size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }
}

#Preview {
    ContentView()
}

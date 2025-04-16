import SwiftUI
import PhotosUI

struct ContentView: View {
    
    @State private var showPhotoPicker = false
    @State private var selectedPhoto = [PhotosPickerItem]()
    
    var body: some View {
        Button(action: {
            self.showPhotoPicker = true
        }) {
            self.addPhotoButton
        }
        .photosPicker(
            isPresented: $showPhotoPicker,
            selection: $selectedPhoto,
            maxSelectionCount: 1,
            matching: .images
        )
    }
    
    private var addPhotoButton: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 15, style: .continuous)
                .frame(width: 250, height: 250)
                .foregroundStyle(.green)
            
            Image(systemName: "plus")
                .font(.largeTitle)
                .foregroundStyle(.white)
        }
    }
}

#Preview {
    ContentView()
}

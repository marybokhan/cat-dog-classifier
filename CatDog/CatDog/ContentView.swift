import SwiftUI

struct ContentView: View {
    var body: some View {
        self.addPhotoButton
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

use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::format::Format;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

fn main() {
    // Setup
    let instance = setup_instance();
    let physical_device = get_physical_device(instance);
    let (device, queues) = get_logical_device(physical_device);
    let queue = get_queue(queues);
    let memory_allocator = create_memory_allocator(device.clone());

    // Image creation
    let height = 1024;
    let width = 1024;
    let image = create_image(memory_allocator.clone(), height, width);

    // Create vertices of single triangle
    let vertex1 = MyVertex { position: [0.0, -0.5] };
    let vertex2 = MyVertex { position: [0.5, 0.5] };
    let vertex3 = MyVertex { position: [-0.5, 0.5] };

    // Create vertex buffer and put the triangle vertices in it
    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    ).expect("Should have been able to create vertex buffer");

    // Create render pass object configured to clear a single image
    let render_pass = create_render_pass(device.clone());

    // Create framebuffer that contains the single image
    let framebuffer = wrap_image_in_framebuffer(image.clone(), render_pass.clone());
}

fn setup_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("No Vulkan library installed");
    let instance = Instance::new(library, InstanceCreateInfo::default())
        .expect("Failed to create Vulkan instance");
    instance
}

fn get_physical_device(instance: Arc<Instance>) -> Arc<PhysicalDevice> {
    let mut physical_devices = instance.enumerate_physical_devices()
        .expect("Couldn't find any physical devices");
    let physical_device = physical_devices.next().expect("No device found");
    physical_device
}

fn get_logical_device(
    physical_device: Arc<PhysicalDevice>,
) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("Couldn't find a graphical queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![
                QueueCreateInfo { queue_family_index, ..Default::default() },
            ],
            ..Default::default()
        },
    ).expect("Unable to create logical device from physical device");
    (device, queues)
}

fn get_queue(mut queues: impl ExactSizeIterator<Item = Arc<Queue>>) -> Arc<Queue> {
    let queue = queues.next().expect("Should have had 1 queue in the iterator of queues");
    queue
}

fn create_memory_allocator(device: Arc<Device>) -> Arc<StandardMemoryAllocator> {
    let allocator = Arc::new(
        StandardMemoryAllocator::new_default(device)
    );
    allocator
}

fn create_image(
    allocator: Arc<StandardMemoryAllocator>,
    height: u32,
    width: u32,
) -> Arc<Image> {
    let image = Image::new(
        allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [height, width, 1],
            usage: ImageUsage::COLOR_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ).expect("Should've been able to create a single 2D image");
    image
}

fn create_render_pass(device: Arc<Device>) -> Arc<RenderPass> {
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: Format::R8G8B8A8_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    ).expect("Should have been able to create render pass");
    render_pass
}

fn wrap_image_in_framebuffer(
    image: Arc<Image>,
    render_pass: Arc<RenderPass>,
) -> Arc<Framebuffer> {
    let view = ImageView::new_default(image)
        .expect("Should have been able to create an image view");
    let framebuffer = Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    ).expect("Should have been able to create framebuffer");
    framebuffer
}

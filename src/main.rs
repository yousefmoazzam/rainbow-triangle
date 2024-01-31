use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::format::Format;

fn main() {
    // Setup
    let instance = setup_instance();
    let physical_device = get_physical_device(instance);
    let (device, queues) = get_logical_device(physical_device);
    let queue = get_queue(queues);
    let memory_allocator = create_memory_allocator(device);

    // Image creation
    let height = 1024;
    let width = 1024;
    let image = create_image(memory_allocator, height, width);
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
        StandardMemoryAllocator::new_default(device.clone())
    );
    allocator
}

fn create_image(
    allocator: Arc<StandardMemoryAllocator>,
    height: u32,
    width: u32,
) -> Arc<Image> {
    let image = Image::new(
        allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [height, width, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ).expect("Should've been able to create a single 2D image");
    image
}

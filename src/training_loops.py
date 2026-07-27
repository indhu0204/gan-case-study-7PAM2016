import torch
from torch import nn


def train_toy_gan(generator, discriminator, data_loader, device, lr=0.001, num_epochs=2000):
    loss_function = nn.BCELoss()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    losses_discriminator = []
    losses_generator = []

    for epoch in range(num_epochs):
        for real_samples, _ in data_loader:
            real_samples = real_samples.to(device)

            real_labels = torch.ones((real_samples.size(0), 1), device=device)
            fake_labels = torch.zeros((real_samples.size(0), 1), device=device)

            latent_space_samples = torch.randn((real_samples.size(0), 2), device=device)
            generated_samples = generator(latent_space_samples)

            all_samples = torch.cat((real_samples, generated_samples), dim=0)
            all_labels = torch.cat((real_labels, fake_labels), dim=0)

            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            latent_space_samples = torch.randn((real_samples.size(0), 2), device=device)

            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)

            loss_generator = loss_function(output_discriminator_generated, real_labels)
            loss_generator.backward()
            optimizer_generator.step()

        losses_discriminator.append(loss_discriminator.item())
        losses_generator.append(loss_generator.item())

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch:4d} | "
                f"Loss D: {loss_discriminator.item():.4f} | "
                f"Loss G: {loss_generator.item():.4f}"
            )

    return losses_discriminator, losses_generator



def train_dcgan_octmnist(generator, discriminator, data_loader, device, latent_dim=100, lr=0.0002, num_epochs=20):
    loss_function = nn.BCELoss()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    losses_discriminator = []
    losses_generator = []

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # Train discriminator
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            discriminator.zero_grad()

            outputs_real = discriminator(real_images)
            loss_real = loss_function(outputs_real, real_labels)

            outputs_fake = discriminator(fake_images.detach())
            loss_fake = loss_function(outputs_fake, fake_labels)

            loss_discriminator = loss_real + loss_fake
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Train generator
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            generator.zero_grad()
            outputs = discriminator(fake_images)
            loss_generator = loss_function(outputs, real_labels)
            loss_generator.backward()
            optimizer_generator.step()

            epoch_d_loss += loss_discriminator.item()
            epoch_g_loss += loss_generator.item()

        losses_discriminator.append(epoch_d_loss / len(data_loader))
        losses_generator.append(epoch_g_loss / len(data_loader))

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Loss D: {losses_discriminator[-1]:.4f} | "
            f"Loss G: {losses_generator[-1]:.4f}"
        )

    return losses_discriminator, losses_generator